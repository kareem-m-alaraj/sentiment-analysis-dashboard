"""
finetune.py
Fine-tunes cardiffnlp/twitter-roberta-base-sentiment-latest into a 2-class
(negative/positive) head on sentiment140 ground truth (posts.raw_label:
0=negative, 4=positive). Reddit rows have raw_label NULL and are excluded
automatically.

Split: stratified, balanced, fixed seed -- 20k train (10k/class), 10k test
(5k/class). The test post_ids are written to `test_split` the first time
this script runs and reused on every run after that, so the held-out set is
fixed forever and training can never draw from it, no matter how many times
the script is interrupted and rerun.

Training: a plain PyTorch loop (no Trainer/accelerate dependency), CPU,
1 epoch, batch 16, max_length 128. Checkpoints every 500 steps under
models/roberta-sentiment140/checkpoints/, so an interrupted run resumes by
reloading the latest checkpoint and skipping ahead in the same deterministic
batch order (optimizer momentum is not preserved across a resume -- an
acceptable simplification for one epoch). The finished model is saved to
models/roberta-sentiment140/.

Usage:
    python src/finetune.py
"""

import random
import time
from pathlib import Path

import psycopg2
import psycopg2.extras
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from classify_sentiment import MODEL_NAME, MAX_TOKENS, DB

# --------------------------------------------------------------------- CONFIG
SEED = 42
TRAIN_PER_CLASS = 10_000
TEST_PER_CLASS = 5_000
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
CHECKPOINT_EVERY = 500
LOG_EVERY = 50

MODEL_DIR = Path("models/roberta-sentiment140")
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
FINETUNED_MODEL_NAME = "roberta-sentiment140-finetuned"   # predictions.model_name key

LABEL2ID = {0: 0, 4: 1}            # raw_label -> training class id
ID2LABEL = {0: "negative", 1: "positive"}

# --------------------------------------------------------------------- DB SETUP
def ensure_test_split_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS test_split (
                post_id    TEXT PRIMARY KEY REFERENCES posts(post_id),
                raw_label  INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT now()
            );
        """)
    conn.commit()


def fetch_labeled_rows(conn):
    """Twitter rows only (raw_label 0/4), ordered so sampling is
    deterministic regardless of physical row order on disk."""
    q = """
        SELECT post_id, text, raw_label
        FROM posts
        WHERE raw_label IS NOT NULL
        ORDER BY post_id
    """
    with conn.cursor() as cur:
        cur.execute(q)
        rows = cur.fetchall()
    by_label = {0: [], 4: []}
    for post_id, text, raw_label in rows:
        by_label[raw_label].append((post_id, text))
    return by_label


def fetch_existing_test_split(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT post_id, raw_label FROM test_split ORDER BY post_id")
        return cur.fetchall()


def write_test_split(conn, rows):
    """rows = list of (post_id, raw_label)"""
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO test_split (post_id, raw_label)
               VALUES %s ON CONFLICT (post_id) DO NOTHING""",
            rows, page_size=1000,
        )
    conn.commit()

# --------------------------------------------------------------------- SPLIT
def get_or_create_test_split(conn, by_label):
    """Returns {post_id: raw_label} for the held-out test set. Built once
    and persisted -- never re-derived, so a rerun can't accidentally shuffle
    a previously-tested post_id into training data."""
    existing = fetch_existing_test_split(conn)
    if existing:
        return dict(existing)

    rng = random.Random(SEED)
    test = {}
    for label in sorted(by_label):
        pool = by_label[label].copy()
        rng.shuffle(pool)
        for post_id, _ in pool[:TEST_PER_CLASS]:
            test[post_id] = label

    write_test_split(conn, list(test.items()))
    return test


def build_train_rows(by_label, test_ids):
    """Stratified, balanced train set drawn from whatever's left after
    excluding test_ids. Fixed seed => same rows every run."""
    rng = random.Random(SEED)
    train = []
    for label in sorted(by_label):
        pool = [(pid, text) for pid, text in by_label[label] if pid not in test_ids]
        rng.shuffle(pool)
        chosen = pool[:TRAIN_PER_CLASS]
        if len(chosen) < TRAIN_PER_CLASS:
            raise RuntimeError(
                f"Only {len(chosen)} rows available for raw_label={label}, "
                f"need {TRAIN_PER_CLASS}. Increase SENTIMENT140_SAMPLE in load_data.py."
            )
        train.extend((pid, text, LABEL2ID[label]) for pid, text in chosen)
    rng.shuffle(train)
    return train

# --------------------------------------------------------------------- MODEL
def find_latest_checkpoint():
    if not CHECKPOINT_DIR.exists():
        return None, 0
    steps = [int(d.name.split("-")[1]) for d in CHECKPOINT_DIR.iterdir()
              if d.is_dir() and d.name.startswith("checkpoint-")]
    if not steps:
        return None, 0
    step = max(steps)
    return CHECKPOINT_DIR / f"checkpoint-{step}", step


def load_model():
    ckpt_dir, resume_step = find_latest_checkpoint()
    source = ckpt_dir if ckpt_dir else MODEL_NAME
    print(f"Loading model from: {source}" +
          (f"  (resuming at step {resume_step:,})" if ckpt_dir else " (fresh, 2-class head)"))

    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=2,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
        ignore_mismatched_sizes=True,
    )
    return tok, model, resume_step


def save_checkpoint(tok, model, step):
    out = CHECKPOINT_DIR / f"checkpoint-{step}"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"  checkpoint saved -> {out}")

# --------------------------------------------------------------------- TRAIN
def train(tok, model, train_rows, resume_step):
    total_steps = len(train_rows) // BATCH_SIZE
    start_idx = resume_step * BATCH_SIZE
    if start_idx >= len(train_rows):
        print("Already completed all steps for this epoch.")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    print(f"Training {total_steps:,} steps total  |  batch={BATCH_SIZE}  |  "
          f"resuming at step {resume_step:,}  |  CPU\n")

    step = resume_step
    t0 = time.time()
    for i in range(start_idx, len(train_rows), BATCH_SIZE):
        chunk = train_rows[i:i + BATCH_SIZE]
        if len(chunk) < BATCH_SIZE:
            break                        # drop the final partial batch
        texts = [r[1] for r in chunk]
        labels = torch.tensor([r[2] for r in chunk])

        enc = tok(texts, padding=True, truncation=True,
                  max_length=MAX_TOKENS, return_tensors="pt")
        optimizer.zero_grad()
        out = model(**enc, labels=labels)
        out.loss.backward()
        optimizer.step()

        step += 1
        if step % LOG_EVERY == 0 or step == total_steps:
            rate = (step - resume_step) / (time.time() - t0)
            print(f"  step {step:,}/{total_steps:,}  loss={out.loss.item():.4f}  "
                  f"({rate:.1f} steps/s)")

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(tok, model, step)

    print(f"\nTraining done in {(time.time()-t0)/60:.1f} min.")

# --------------------------------------------------------------------- MAIN
def main():
    conn = psycopg2.connect(**DB)
    ensure_test_split_table(conn)

    by_label = fetch_labeled_rows(conn)
    test_ids_map = get_or_create_test_split(conn, by_label)
    print(f"Test split: {len(test_ids_map):,} rows "
          f"({sum(1 for v in test_ids_map.values() if v == 0):,} neg / "
          f"{sum(1 for v in test_ids_map.values() if v == 4):,} pos) "
          f"-- held out, never trained on")

    train_rows = build_train_rows(by_label, set(test_ids_map))
    print(f"Train split: {len(train_rows):,} rows (balanced 2-class)\n")

    tok, model, resume_step = load_model()
    train(tok, model, train_rows, resume_step)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    print(f"\nFinal model saved -> {MODEL_DIR}")

    conn.close()


if __name__ == "__main__":
    main()
