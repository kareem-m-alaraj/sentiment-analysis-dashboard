"""
evaluate.py
Scores a sentiment model against sentiment140 ground truth (posts.raw_label:
0=negative, 4=positive), restricted to the fixed held-out test set written
by finetune.py to `test_split`. Reddit rows have raw_label NULL and never
appear here.

--model zeroshot (default) -- cardiffnlp/twitter-roberta-base-sentiment-latest,
    the pretrained 3-class (neg/neu/pos) checkpoint. Reports both views:
      STRICT -- score only rows the model called negative/positive; neutral
                predictions count as abstentions. Reports accuracy + coverage.
      FORCED -- every row gets a 2-way call. Rows already predicted
                negative/positive keep that call. Neutral rows are re-run
                picking whichever of P(neg)/P(pos) is higher (predictions
                only stores the winning label, not full probabilities, so
                this re-run is unavoidable -- cached in `forced_predictions`
                so it only happens once).

--model finetuned -- the 2-class model fine-tuned by finetune.py, loaded
    from models/roberta-sentiment140/. Natively 2-way, so a single report.

Both modes cache their predictions in `predictions` (keyed by model_name),
so rerunning evaluate.py is cheap after the first pass.

Usage:
    python src/evaluate.py                    # zero-shot (default)
    python src/evaluate.py --model finetuned   # fine-tuned model
"""

import argparse
import time
from pathlib import Path

import psycopg2
import psycopg2.extras
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from classify_sentiment import (
    MODEL_NAME, BATCH_SIZE, MAX_TOKENS, DB,
    load_model, classify_texts, ensure_predictions_table, write_batch,
)

TARGET_ACCURACY = 0.85
RAW_LABEL_TO_GROUND_TRUTH = {0: "negative", 4: "positive"}

FINETUNED_MODEL_NAME = "roberta-sentiment140-finetuned"
FINETUNED_MODEL_DIR = Path("models/roberta-sentiment140")

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


def test_split_exists(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('test_split')")
        return cur.fetchone()[0] is not None


def fetch_test_split(conn):
    """Held-out rows written by finetune.py: (post_id, text, raw_label)."""
    q = """
        SELECT p.post_id, p.text, p.raw_label
        FROM posts p
        JOIN test_split t ON t.post_id = p.post_id
        ORDER BY p.post_id
    """
    with conn.cursor() as cur:
        cur.execute(q)
        return cur.fetchall()


def fetch_predictions(conn, model_name, post_ids):
    q = """
        SELECT post_id, label, score FROM predictions
        WHERE model_name = %s AND post_id = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(q, [model_name, list(post_ids)])
        return {pid: (label, score) for pid, label, score in cur.fetchall()}


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
               VALUES %s ON CONFLICT (post_id, model_name) DO NOTHING""",
            rows, page_size=1000,
        )
    conn.commit()

# --------------------------------------------------------------------- INFERENCE
def ensure_classified(conn, model_name, rows, classify_fn, tok, model, desc):
    """rows: list of (post_id, text). Classifies any post_id missing from
    `predictions` for model_name, writes results, returns the full
    {post_id: (label, score)} map covering all of rows."""
    ids = [pid for pid, _ in rows]
    have = fetch_predictions(conn, model_name, ids)
    missing = [(pid, text) for pid, text in rows if pid not in have]
    if not missing:
        return have

    print(f"{desc}: classifying {len(missing):,} test rows not yet in predictions cache")
    total = len(missing)
    done = 0
    t0 = time.time()
    for i in range(0, total, BATCH_SIZE):
        chunk = missing[i:i + BATCH_SIZE]
        ids_chunk = [r[0] for r in chunk]
        texts_chunk = [r[1] for r in chunk]
        preds = classify_fn(texts_chunk, tok, model)
        write_batch(conn, [(pid, model_name, label, score)
                           for pid, (label, score) in zip(ids_chunk, preds)])
        have.update({pid: (label, score) for pid, (label, score) in zip(ids_chunk, preds)})
        done += len(chunk)
        if done % (BATCH_SIZE * 10) == 0 or done == total:
            rate = done / (time.time() - t0)
            print(f"  {done:,}/{total:,}  ({rate:.0f} rows/s)")
    return have


@torch.no_grad()
def classify_forced(texts, tok, model):
    """Force a neg/pos call on the 3-class zero-shot model, ignoring neutral."""
    enc = tok(texts, padding=True, truncation=True,
              max_length=MAX_TOKENS, return_tensors="pt")
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    neg, pos = probs[:, 0].tolist(), probs[:, 2].tolist()
    return [("positive", round(p, 4)) if p >= n else ("negative", round(n, 4))
            for n, p in zip(neg, pos)]


def compute_forced_labels(conn, rows):
    """rows: (post_id, text, raw_label, label, score) with 3-class zero-shot
    predictions. Returns dict post_id -> forced 2-way label."""
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

# ------------------------------------------------------------ MODEL (fine-tuned)
def load_finetuned_model():
    if not FINETUNED_MODEL_DIR.exists():
        raise SystemExit(f"No fine-tuned model found at {FINETUNED_MODEL_DIR}. "
                          f"Run src/finetune.py first.")
    tok = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_DIR)
    model.eval()
    return tok, model


@torch.no_grad()
def classify_finetuned(texts, tok, model):
    """The fine-tuned model is natively 2-class; no forced/strict split needed."""
    enc = tok(texts, padding=True, truncation=True,
              max_length=MAX_TOKENS, return_tensors="pt")
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    conf, idx = torch.max(probs, dim=-1)
    return [(model.config.id2label[i.item()], round(c.item(), 4))
            for i, c in zip(idx, conf)]

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["zeroshot", "finetuned"], default="zeroshot",
                    help="which model to score against the held-out test set")
    args = ap.parse_args()

    conn = psycopg2.connect(**DB)
    ensure_predictions_table(conn)

    if not test_split_exists(conn):
        print("No test_split table found. Run src/finetune.py first to "
              "create the held-out test set.")
        conn.close()
        return

    test_rows = fetch_test_split(conn)
    total_labeled = len(test_rows)
    if total_labeled == 0:
        print("test_split is empty. Run src/finetune.py first.")
        conn.close()
        return

    print(f"Held-out test set: {total_labeled:,} rows")
    text_rows = [(pid, text) for pid, text, _ in test_rows]

    if args.model == "zeroshot":
        tok, model = load_model()
        preds = ensure_classified(conn, MODEL_NAME, text_rows,
                                   classify_texts, tok, model, "zero-shot")
        rows = [(pid, text, raw_label, *preds[pid]) for pid, text, raw_label in test_rows]

        strict_pairs = [(RAW_LABEL_TO_GROUND_TRUTH[raw_label], label)
                        for _, _, raw_label, label, _ in rows
                        if label in ("negative", "positive")]
        print_report("ZERO-SHOT STRICT (neutral = abstain)", strict_pairs, total_labeled)

        forced_labels = compute_forced_labels(conn, rows)
        forced_pairs = [(RAW_LABEL_TO_GROUND_TRUTH[raw_label], forced_labels[post_id])
                        for post_id, _, raw_label, _, _ in rows]
        print_report("ZERO-SHOT FORCED (neutral re-run, 2-way)", forced_pairs, total_labeled)

    else:
        tok, model = load_finetuned_model()
        preds = ensure_classified(conn, FINETUNED_MODEL_NAME, text_rows,
                                   classify_finetuned, tok, model, "fine-tuned")
        pairs = [(RAW_LABEL_TO_GROUND_TRUTH[raw_label], preds[pid][0])
                for pid, _, raw_label in test_rows]
        print_report("FINE-TUNED (2-way)", pairs, total_labeled)

    conn.close()


if __name__ == "__main__":
    main()
