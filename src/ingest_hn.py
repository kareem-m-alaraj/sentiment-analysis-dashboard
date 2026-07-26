"""
ingest_hn.py
Pulls current front-page-eligible stories from the HN Firebase API
(https://hacker-news.firebaseio.com/v0/, no auth) and inserts them into the
existing `posts` table alongside reddit/twitter rows.

Usage:
    python src/ingest_hn.py                # fetch, insert, report counts
    python src/ingest_hn.py --limit 50     # only process the first 50 story ids
    python src/ingest_hn.py --dry-run      # fetch + build rows, no DB write

Does not classify — classify_sentiment.py's unclassified-row query picks up
hackernews posts the same way it picks up reddit/twitter ones.
"""

import argparse
import html
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import psycopg2
import requests
from psycopg2.extras import execute_values
from dotenv import load_dotenv

from topics import tag_topic

# --------------------------------------------------------------------- CONFIG
API_BASE = "https://hacker-news.firebaseio.com/v0"

MAX_WORKERS = 15
FETCH_TIMEOUT = 10       # seconds per request
FETCH_RETRIES = 2        # additional attempts after the first

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
    """Returns (rows, skipped_filtered, skipped_empty). rows match the posts
    schema: (post_id, source, topic, text, created_at, raw_label)."""
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
        rows.append((
            f"hn_{item['id']}",
            "hackernews",
            tag_topic(text),
            text,
            created_at,
            None,
        ))
    return rows, skipped_filtered, skipped_empty

# ------------------------------------------------------------------ DB WRITE
def insert_rows(rows) -> int:
    """Inserts rows, ON CONFLICT DO NOTHING. Returns count actually inserted."""
    load_dotenv()
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
    )
    with conn, conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO posts (post_id, source, topic, text, created_at, raw_label) "
            "VALUES %s ON CONFLICT (post_id) DO NOTHING",
            rows, page_size=len(rows),   # single page => rowcount is accurate
        )
        inserted = cur.rowcount
    conn.close()
    return inserted

# --------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N new story ids")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and build rows, no DB write")
    args = ap.parse_args()

    print("Fetching new story ids...")
    ids = fetch_new_story_ids(args.limit)
    print(f"  {len(ids)} ids to fetch\n")

    print(f"Fetching item details ({MAX_WORKERS} workers)...")
    items, failed = fetch_items_concurrently(ids)
    print(f"  {len(items)} fetched, {failed} failed (skipped)\n")

    rows, skipped_filtered, skipped_empty = build_rows(items)
    print(f"  {len(rows)} rows built "
          f"({skipped_filtered} skipped non-story/dead/deleted, "
          f"{skipped_empty} skipped empty text)\n")

    if not rows:
        print("Nothing to insert.")
        return

    if args.dry_run:
        print(f"[dry-run] would attempt to insert {len(rows)} rows. Sample:")
        for r in rows[:5]:
            print(f"  {r[0]:<12} [{r[2]:<10}] {r[3][:70]}")
        return

    inserted = insert_rows(rows)
    duplicates = len(rows) - inserted
    print(f"Inserted {inserted:,}, {duplicates:,} duplicates skipped "
          f"(already in posts).")


if __name__ == "__main__":
    main()
