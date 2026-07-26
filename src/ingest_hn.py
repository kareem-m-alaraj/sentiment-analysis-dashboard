"""
ingest_hn.py
Pulls current front-page-eligible stories from the HN Firebase API
(https://hacker-news.firebaseio.com/v0/, no auth) and writes them to either
the `posts` table or a local parquet partition.

Usage:
    python src/ingest_hn.py                          # fetch, insert into postgres
    python src/ingest_hn.py --sink parquet            # fetch, write data/live/hn/<date>.parquet
    python src/ingest_hn.py --limit 50                # only process the first 50 story ids
    python src/ingest_hn.py --dry-run                 # fetch + build rows, no write

Fetching/cleaning/tagging (fetch_hn_rows) has no DB dependency -- only
insert_rows() touches Postgres, and only when --sink postgres is used (the
default, so existing behaviour is unchanged). --sink parquet never imports
psycopg2 and never touches Postgres, so it can run in environments (e.g. a
scheduled Action) that have no database access.

--sink postgres does not classify -- classify_sentiment.py's unclassified-row
query picks up hackernews posts the same way it picks up reddit/twitter ones,
once they're in `posts`. --sink parquet classifies inline via
sentiment_model.classify_texts, because its output must match
data/processed/dashboard.parquet's columns (posts JOIN predictions), which
the dashboard reads directly and has no separate predictions table to defer
to.
"""

import argparse
import html
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from topics import tag_topic

# --------------------------------------------------------------------- CONFIG
API_BASE = "https://hacker-news.firebaseio.com/v0"

MAX_WORKERS = 15
FETCH_TIMEOUT = 10       # seconds per request
FETCH_RETRIES = 2        # additional attempts after the first

LIVE_DIR = Path("data/live/hn")
DEDUPE_LOOKBACK_DAYS = 2  # check this many prior daily partitions for dupes

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

# ------------------------------------------------------------------- FETCHING
def fetch_new_story_ids(limit=None):
    resp = requests.get(f"{API_BASE}/newstories.json", timeout=FETCH_TIMEOUT)
    resp.raise_for_status()
    ids = resp.json()
    return ids[:limit] if limit else ids


def fetch_item(item_id):
    """GET one item, retrying on error. Returns the item dict, or None if it
    never succeeded (caller skips it rather than aborting the run)."""
    url = f"{API_BASE}/item/{item_id}.json"
    for attempt in range(FETCH_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=FETCH_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt < FETCH_RETRIES:
                time.sleep(0.5 * (attempt + 1))
    return None


def fetch_items_concurrently(ids):
    items, failed = [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_item, id_) for id_ in ids]
        for fut in as_completed(futures):
            item = fut.result()
            if item is None:
                failed += 1
            else:
                items.append(item)
    return items, failed

# --------------------------------------------------------------------- CLEAN
def clean_html(raw: str) -> str:
    text = html.unescape(raw or "")
    text = TAG_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()

# ------------------------------------------------------------------ ROW BUILD
def build_rows(items):
    """Returns (rows, skipped_filtered, skipped_empty). rows are dicts
    matching the posts schema: post_id, source, topic, text, created_at,
    raw_label."""
    rows, skipped_filtered, skipped_empty = [], 0, 0
    for item in items:
        if item.get("type") != "story" or item.get("dead") or item.get("deleted"):
            skipped_filtered += 1
            continue

        title = html.unescape(item.get("title") or "").strip()
        body = clean_html(item.get("text") or "")
        text = f"{title} {body}".strip() if body else title
        if not text:
            skipped_empty += 1
            continue

        created_at = (
            datetime.fromtimestamp(item["time"], tz=timezone.utc).replace(tzinfo=None)
            if item.get("time") else None
        )
        rows.append({
            "post_id":    f"hn_{item['id']}",
            "source":     "hackernews",
            "topic":      tag_topic(text),
            "text":       text,
            "created_at": created_at,
            "raw_label":  None,
        })
    return rows, skipped_filtered, skipped_empty


def fetch_hn_rows(limit=None):
    """Fetch, clean, and topic-tag current HN stories. No DB dependency --
    returns (rows, stats). rows are dicts, ready for either sink."""
    print("Fetching new story ids...")
    ids = fetch_new_story_ids(limit)
    print(f"  {len(ids)} ids to fetch\n")

    print(f"Fetching item details ({MAX_WORKERS} workers)...")
    items, failed = fetch_items_concurrently(ids)
    print(f"  {len(items)} fetched, {failed} failed (skipped)\n")

    rows, skipped_filtered, skipped_empty = build_rows(items)
    print(f"  {len(rows)} rows built "
          f"({skipped_filtered} skipped non-story/dead/deleted, "
          f"{skipped_empty} skipped empty text)\n")

    stats = {
        "ids": len(ids), "fetched": len(items), "failed": failed,
        "skipped_filtered": skipped_filtered, "skipped_empty": skipped_empty,
    }
    return rows, stats

# ------------------------------------------------------------------ DB SINK
def insert_rows(rows) -> int:
    """Inserts rows into Postgres, ON CONFLICT DO NOTHING. Returns count
    actually inserted. Only sink that touches Postgres."""
    import psycopg2
    from psycopg2.extras import execute_values

    load_dotenv()
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
    )
    cols = ["post_id", "source", "topic", "text", "created_at", "raw_label"]
    records = [tuple(r[c] for c in cols) for r in rows]
    with conn, conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO posts (post_id, source, topic, text, created_at, raw_label) "
            "VALUES %s ON CONFLICT (post_id) DO NOTHING",
            records, page_size=len(records),   # single page => rowcount is accurate
        )
        inserted = cur.rowcount
    conn.close()
    return inserted

# -------------------------------------------------------------- PARQUET SINK
# Matches data/processed/dashboard.parquet's columns (posts JOIN predictions),
# not the posts table itself -- no raw_label, but with model_name/label/score
# so live rows can be concatenated straight into the dashboard's data.
PARQUET_COLUMNS = ["post_id", "source", "topic", "text", "created_at",
                   "model_name", "label", "score"]


def write_parquet_sink(rows):
    """Classifies rows, then writes today's UTC-dated partition to LIVE_DIR:
    merges with today's existing partition if present (so repeat same-day
    runs accumulate rather than overwrite), then drops anything already seen
    in the prior DEDUPE_LOOKBACK_DAYS partitions. Never touches Postgres.
    Returns (written, deduped, out_path)."""
    import pandas as pd
    from sentiment_model import MODEL_NAME, classify_texts

    preds = classify_texts([r["text"] for r in rows])
    for r, (label, score) in zip(rows, preds):
        r["model_name"] = MODEL_NAME
        r["label"] = label
        r["score"] = score

    today = datetime.now(timezone.utc).date()
    out_path = LIVE_DIR / f"{today.isoformat()}.parquet"

    df = pd.DataFrame(rows, columns=PARQUET_COLUMNS)
    if out_path.exists():
        df = pd.concat([pd.read_parquet(out_path), df], ignore_index=True)
        df = df.drop_duplicates(subset="post_id", keep="last")

    existing_ids = set()
    for days_back in range(1, DEDUPE_LOOKBACK_DAYS + 1):
        prior = LIVE_DIR / f"{(today - timedelta(days=days_back)).isoformat()}.parquet"
        if prior.exists():
            existing_ids.update(pd.read_parquet(prior, columns=["post_id"])["post_id"])

    before = len(df)
    df = df[~df["post_id"].isin(existing_ids)].reset_index(drop=True)
    deduped = before - len(df)

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df), deduped, out_path

# --------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N new story ids")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and build rows, no write")
    ap.add_argument("--sink", choices=["postgres", "parquet"], default="postgres",
                    help="where rows are written (default: postgres)")
    args = ap.parse_args()

    rows, _ = fetch_hn_rows(args.limit)

    if not rows:
        print("Nothing to write.")
        return

    if args.dry_run:
        print(f"[dry-run] would write {len(rows)} rows via --sink={args.sink}. Sample:")
        for r in rows[:5]:
            print(f"  {r['post_id']:<12} [{r['topic']:<10}] {r['text'][:70]}")
        return

    if args.sink == "postgres":
        inserted = insert_rows(rows)
        duplicates = len(rows) - inserted
        print(f"Inserted {inserted:,}, {duplicates:,} duplicates skipped "
              f"(already in posts).")
    else:
        written, deduped, out_path = write_parquet_sink(rows)
        print(f"Wrote {written:,} rows -> {out_path} "
              f"({deduped:,} deduped against last {DEDUPE_LOOKBACK_DAYS} days).")


if __name__ == "__main__":
    main()
