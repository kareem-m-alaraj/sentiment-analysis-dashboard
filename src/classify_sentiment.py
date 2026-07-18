"""
classify_sentiment.py
Runs cardiffnlp/twitter-roberta-base-sentiment-latest over rows in `posts`,
writes results to `predictions`. Resumable: skips already-classified post_ids.

Usage:
    python src/classify_sentiment.py --validate      # 20 rows, prints, no DB write
    python src/classify_sentiment.py --limit 2000    # first N unclassified rows
    python src/classify_sentiment.py                 # full remaining pass
"""

import os
import sys
import argparse
import time

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------------------------- CONFIG
load_dotenv()

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# model's own label order for this checkpoint: 0=neg, 1=neu, 2=pos
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

BATCH_SIZE = 32          # rows per forward pass; safe for CPU RAM
MAX_TOKENS = 128         # tweets/reddit titles are short; 128 is plenty
DB = dict(
    dbname=os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    host=os.environ["DB_HOST"],
    port=os.environ["DB_PORT"],
)

# --------------------------------------------------------------------- DB SETUP
def ensure_predictions_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                post_id      TEXT NOT NULL REFERENCES posts(post_id),
                model_name   TEXT NOT NULL,
                label        TEXT NOT NULL,
                score        REAL NOT NULL,
                predicted_at TIMESTAMP DEFAULT now(),
                PRIMARY KEY (post_id, model_name)
            );
        """)
    conn.commit()


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
    params = [MODEL_NAME]
    if limit:
        q += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(q, params)
        return cur.fetchall()          # list of (post_id, text)


def write_batch(conn, rows):
    """rows = list of (post_id, model_name, label, score)"""
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO predictions (post_id, model_name, label, score)
               VALUES %s
               ON CONFLICT (post_id, model_name) DO NOTHING""",
            rows, page_size=1000,
        )
    conn.commit()

# --------------------------------------------------------------------- MODEL
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
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
    return [(ID2LABEL[i.item()], round(c.item(), 4))
            for i, c in zip(idx, conf)]

# --------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="classify 20 rows, print results, no DB write")
    ap.add_argument("--limit", type=int, default=None,
                    help="only classify first N unclassified rows")
    args = ap.parse_args()

    print(f"Loading model: {MODEL_NAME}")
    print("(first run downloads ~500MB — one time, then cached)\n")
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
        batch = [(pid, MODEL_NAME, label, score)
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