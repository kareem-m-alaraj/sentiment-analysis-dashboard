"""
evaluate.py
Scores cardiffnlp/twitter-roberta-base-sentiment-latest predictions against
sentiment140 ground truth. Only twitter rows carry a label (posts.raw_label:
0=negative, 4=positive) -- reddit rows have raw_label NULL and are excluded
by the ground-truth filter itself, no source check needed.

The model has 3 classes (neg/neu/pos) but ground truth is 2-way, so this
reports both views:

  STRICT  -- score only rows the model called negative/positive; neutral
             predictions count as abstentions. Reports accuracy + coverage.
  FORCED  -- every labeled row gets a 2-way call. Rows already predicted
             negative/positive keep that call (whichever of neg/pos beat
             the other also beat neutral, so it's still the 2-way winner).
             Neutral rows are re-run picking whichever of P(neg)/P(pos) is
             higher. predictions only stores the winning label, not full
             probabilities, so this re-run is unavoidable -- results are
             cached in `forced_predictions` so it only happens once.

Usage:
    python src/evaluate.py
"""

import time

import psycopg2
import psycopg2.extras
import torch

from classify_sentiment import MODEL_NAME, BATCH_SIZE, MAX_TOKENS, DB, load_model

TARGET_ACCURACY = 0.85
RAW_LABEL_TO_GROUND_TRUTH = {0: "negative", 4: "positive"}

# --------------------------------------------------------------------- DB SETUP
def ensure_forced_predictions_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forced_predictions (
                post_id      TEXT NOT NULL REFERENCES posts(post_id),
                model_name   TEXT NOT NULL,
                label        TEXT NOT NULL,
                score        REAL NOT NULL,
                predicted_at TIMESTAMP DEFAULT now(),
                PRIMARY KEY (post_id, model_name)
            );
        """)
    conn.commit()


def fetch_labeled_predictions(conn):
    """Twitter rows only (raw_label 0/4) with their roberta prediction."""
    q = """
        SELECT p.post_id, p.text, p.raw_label, pr.label, pr.score
        FROM posts p
        JOIN predictions pr
          ON pr.post_id = p.post_id AND pr.model_name = %s
        WHERE p.raw_label IN (0, 4)
    """
    with conn.cursor() as cur:
        cur.execute(q, [MODEL_NAME])
        return cur.fetchall()   # (post_id, text, raw_label, label, score)


def fetch_cached_forced(conn, post_ids):
    if not post_ids:
        return {}
    q = """
        SELECT post_id, label FROM forced_predictions
        WHERE model_name = %s AND post_id = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(q, [MODEL_NAME, list(post_ids)])
        return dict(cur.fetchall())


def write_forced_batch(conn, rows):
    """rows = list of (post_id, model_name, label, score)"""
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO forced_predictions (post_id, model_name, label, score)
               VALUES %s
               ON CONFLICT (post_id, model_name) DO NOTHING""",
            rows, page_size=1000,
        )
    conn.commit()

# ------------------------------------------------------------------ MODEL (2-way)
@torch.no_grad()
def classify_forced(texts, tok, model):
    """Force a neg/pos call, ignoring the neutral class entirely."""
    enc = tok(texts, padding=True, truncation=True,
              max_length=MAX_TOKENS, return_tensors="pt")
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    neg, pos = probs[:, 0].tolist(), probs[:, 2].tolist()
    return [("positive", round(p, 4)) if p >= n else ("negative", round(n, 4))
            for n, p in zip(neg, pos)]


def compute_forced_labels(conn, rows):
    """rows: (post_id, text, raw_label, label, score) from fetch_labeled_predictions.
    Returns dict post_id -> forced 2-way label."""
    forced = {pid: label for pid, _, _, label, _ in rows if label != "neutral"}

    neutral_rows = [(pid, text) for pid, text, _, label, _ in rows if label == "neutral"]
    if not neutral_rows:
        return forced

    ensure_forced_predictions_table(conn)
    neutral_ids = [pid for pid, _ in neutral_rows]
    cached = fetch_cached_forced(conn, neutral_ids)
    forced.update(cached)

    remaining = [(pid, text) for pid, text in neutral_rows if pid not in cached]
    if not remaining:
        return forced

    print(f"Forced pass: re-running model on {len(remaining):,} neutral rows "
          f"(already cached: {len(cached):,})")
    tok, model = load_model()
    total = len(remaining)
    done = 0
    t0 = time.time()
    for i in range(0, total, BATCH_SIZE):
        chunk = remaining[i:i + BATCH_SIZE]
        ids = [r[0] for r in chunk]
        texts = [r[1] for r in chunk]
        preds = classify_forced(texts, tok, model)
        write_forced_batch(conn, [(pid, MODEL_NAME, label, score)
                                   for pid, (label, score) in zip(ids, preds)])
        for pid, (label, _) in zip(ids, preds):
            forced[pid] = label
        done += len(chunk)

        if done % (BATCH_SIZE * 10) == 0 or done == total:
            rate = done / (time.time() - t0)
            print(f"  {done:,}/{total:,}  ({rate:.0f} rows/s)")

    return forced

# --------------------------------------------------------------------- METRICS
def confusion_counts(pairs):
    """pairs: list of (truth, pred), each 'negative' | 'positive'."""
    tp = sum(1 for t, p in pairs if t == "positive" and p == "positive")
    tn = sum(1 for t, p in pairs if t == "negative" and p == "negative")
    fp = sum(1 for t, p in pairs if t == "negative" and p == "positive")
    fn = sum(1 for t, p in pairs if t == "positive" and p == "negative")
    return tp, tn, fp, fn


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def print_report(name, pairs, total_labeled):
    tp, tn, fp, fn = confusion_counts(pairs)
    scored = len(pairs)
    accuracy = (tp + tn) / scored if scored else 0.0
    coverage = scored / total_labeled if total_labeled else 0.0
    verdict = "meets" if accuracy >= TARGET_ACCURACY else "below"

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy:.4f}  ({tp + tn:,}/{scored:,})  "
          f"[{verdict} {TARGET_ACCURACY:.0%} target]")
    print(f"Coverage:  {coverage:.4f}  ({scored:,}/{total_labeled:,})")

    print("\nConfusion matrix (rows=truth, cols=predicted):")
    print(f"{'':>12}{'pred_neg':>10}{'pred_pos':>10}")
    print(f"{'true_neg':>12}{tn:>10}{fp:>10}")
    print(f"{'true_pos':>12}{fn:>10}{tp:>10}")

    p_pos, r_pos, f_pos = precision_recall_f1(tp, fp, fn)
    p_neg, r_neg, f_neg = precision_recall_f1(tn, fn, fp)
    print("\nPer-class precision / recall / f1:")
    print(f"{'positive':<10} P={p_pos:.4f}  R={r_pos:.4f}  F1={f_pos:.4f}")
    print(f"{'negative':<10} P={p_neg:.4f}  R={r_neg:.4f}  F1={f_neg:.4f}")

# --------------------------------------------------------------------- MAIN
def main():
    conn = psycopg2.connect(**DB)

    rows = fetch_labeled_predictions(conn)
    total_labeled = len(rows)
    if total_labeled == 0:
        print("No labeled (twitter) rows with predictions found. "
              "Run classify_sentiment.py first.")
        conn.close()
        return

    print(f"Labeled twitter rows with {MODEL_NAME} predictions: {total_labeled:,}")

    strict_pairs = [
        (RAW_LABEL_TO_GROUND_TRUTH[raw_label], label)
        for _, _, raw_label, label, _ in rows
        if label in ("negative", "positive")
    ]
    print_report("STRICT (neutral = abstain)", strict_pairs, total_labeled)

    forced_labels = compute_forced_labels(conn, rows)
    forced_pairs = [
        (RAW_LABEL_TO_GROUND_TRUTH[raw_label], forced_labels[post_id])
        for post_id, _, raw_label, _, _ in rows
    ]
    print_report("FORCED (neutral re-run, 2-way)", forced_pairs, total_labeled)

    conn.close()


if __name__ == "__main__":
    main()
