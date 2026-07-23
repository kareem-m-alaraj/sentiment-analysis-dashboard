"""
classify_finetuned.py
Runs the fine-tuned model (models/roberta-sentiment140/, trained by
finetune.py) over ALL rows in `posts` -- reddit and twitter alike -- writing
results to `predictions` under its own model_name. Same resumable pattern as
classify_sentiment.py: skips post_ids that already have a row for this
model_name, commits per batch so a full pass can be interrupted and resumed.

Unlike classify_sentiment.py's 3-class checkpoint, this model is natively
2-class (negative/positive) -- no neutral bucket, no strict/forced split.

Usage:
    python src/classify_finetuned.py --validate      # 20 rows, prints, no DB write
    python src/classify_finetuned.py --limit 2000    # first N unclassified rows
    python src/classify_finetuned.py                 # full remaining pass
"""

import argparse
import time

import psycopg2
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from classify_sentiment import BATCH_SIZE, MAX_TOKENS, DB, ensure_predictions_table, write_batch
from finetune import FINETUNED_MODEL_NAME, MODEL_DIR

# --------------------------------------------------------------------- DB
def fetch_unclassified(conn, limit=None):
    """Rows in posts with no prediction for THIS model yet."""
    q = """
        SELECT p.post_id, p.text
        FROM posts p
        LEFT JOIN predictions pr
          ON pr.post_id = p.post_id AND pr.model_name = %s
        WHERE pr.post_id IS NULL
          AND length(trim(p.text)) > 0
    """
    params = [FINETUNED_MODEL_NAME]
    if limit:
        q += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(q, params)
        return cur.fetchall()          # list of (post_id, text)

# --------------------------------------------------------------------- MODEL
def load_model():
    if not MODEL_DIR.exists():
        raise SystemExit(f"No fine-tuned model found at {MODEL_DIR}. Run src/finetune.py first.")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tok, model


@torch.no_grad()
def classify_texts(texts, tok, model):
    """Returns list of (label, score) aligned with input texts."""
    enc = tok(texts, padding=True, truncation=True,
              max_length=MAX_TOKENS, return_tensors="pt")
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    conf, idx = torch.max(probs, dim=-1)
    return [(model.config.id2label[i.item()], round(c.item(), 4))
            for i, c in zip(idx, conf)]

# --------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="classify 20 rows, print results, no DB write")
    ap.add_argument("--limit", type=int, default=None,
                    help="only classify first N unclassified rows")
    args = ap.parse_args()

    print(f"Loading fine-tuned model: {MODEL_DIR}\n")
    tok, model = load_model()

    conn = psycopg2.connect(**DB)
    ensure_predictions_table(conn)

    # ---- validate mode: small slice, print, no write ----
    if args.validate:
        rows = fetch_unclassified(conn, limit=20)
        if not rows:
            print("Nothing unclassified — all rows already have predictions.")
            conn.close()
            return
        ids, texts = zip(*rows)
        preds = classify_texts(list(texts), tok, model)
        print(f"{'LABEL':<9} {'SCORE':<7} TEXT")
        print("-" * 80)
        for (label, score), txt in zip(preds, texts):
            print(f"{label:<9} {score:<7} {txt[:60]}")
        print("\nValidate only — nothing written. "
              "If labels look sane, run without --validate.")
        conn.close()
        return

    # ---- full / limited pass ----
    rows = fetch_unclassified(conn, limit=args.limit)
    total = len(rows)
    if total == 0:
        print("Nothing to do — every row already classified for this model.")
        conn.close()
        return

    print(f"Classifying {total:,} rows  |  batch={BATCH_SIZE}  |  CPU\n")
    done = 0
    t0 = time.time()
    for i in range(0, total, BATCH_SIZE):
        chunk = rows[i:i + BATCH_SIZE]
        ids = [r[0] for r in chunk]
        texts = [r[1] for r in chunk]
        preds = classify_texts(texts, tok, model)
        batch = [(pid, FINETUNED_MODEL_NAME, label, score)
                 for pid, (label, score) in zip(ids, preds)]
        write_batch(conn, batch)          # commit per batch = resumable
        done += len(chunk)

        if done % (BATCH_SIZE * 10) == 0 or done == total:
            rate = done / (time.time() - t0)
            eta = (total - done) / rate if rate else 0
            print(f"  {done:,}/{total:,}  ({rate:.0f} rows/s, ETA {eta/60:.1f} min)")

    print(f"\nDone. {done:,} rows classified in {(time.time()-t0)/60:.1f} min.")
    conn.close()


if __name__ == "__main__":
    main()
