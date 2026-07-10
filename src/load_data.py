"""
load_data.py — Project 1 (Sentiment Analysis Dashboard)

Normalizes the three raw CSVs into the unified `posts` schema, writes an
inspectable processed CSV, then loads it into the Postgres `posts` table.

Schema (matches DB table `posts`):
    post_id    TEXT (PK)  source-prefixed, globally unique
    source     TEXT       'reddit' | 'twitter'
    topic      TEXT       keyword-tagged, fallback 'general'
    text       TEXT       reddit: title (fallback body); twitter: tweet
    created_at TIMESTAMP  parsed; unparseable -> NULL
    raw_label  INTEGER    twitter: 0/4; reddit: NULL
"""

import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# ---------------------------------------------------------------- CONFIG ----
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_CSV    = PROCESSED_DIR / "posts_unified.csv"

# sentiment140 is 1.6M rows. None = load all; an int = sample that many.
# Default 50k: plenty as a labeled eval set, fast to load and inspect.
SENTIMENT140_SAMPLE = 50_000

# Which reddit column holds the readable date. If reddit created_at comes back
# mostly NULL after a run (diagnostics print this), flip to "created" and re-run.
REDDIT_DATE_COL = "timestamp"

# ------------------------------------------------------------ TOPIC TAGGING -
TOPIC_KEYWORDS = {
    "android":    ["android", "pixel", "samsung", "galaxy"],
    "nvidia":     ["nvidia", "geforce", "rtx", "cuda"],
    "bmw":        ["bmw", "m3", "m5", "bimmer"],
    "investing":  ["invest", "stock", "market", "shares", "portfolio", "wsb"],
    "technology": ["tech", "software", "hardware", "chip", "ai", "app"],
}

def tag_topic(text: str) -> str:
    t = (text or "").lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return topic
    return "general"

# ------------------------------------------------------------------ LOADERS -
def load_reddit(filename: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / filename)
    title = df["title"].fillna("").astype(str)
    body  = df.get("body", pd.Series([""] * len(df))).fillna("").astype(str)
    text  = title.where(title.str.strip() != "", body)

    out = pd.DataFrame({
        "post_id":    "reddit_" + df["id"].astype(str),
        "source":     "reddit",
        "text":       text,
        "created_at": pd.to_datetime(df[REDDIT_DATE_COL], errors="coerce"),
        "raw_label":  pd.NA,
    })
    out["topic"] = out["text"].apply(tag_topic)
    return out

def load_sentiment140() -> pd.DataFrame:
    cols = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(RAW_DIR / "sentiment140.csv",
                     encoding="latin-1", header=None, names=cols)
    if SENTIMENT140_SAMPLE and SENTIMENT140_SAMPLE < len(df):
        df = df.sample(n=SENTIMENT140_SAMPLE, random_state=42)

    # 'Mon Apr 06 22:19:45 PDT 2009' -> strip the tz token, then parse
    cleaned = df["date"].str.replace(r"\s[A-Z]{3}\s", " ", regex=True)
    created = pd.to_datetime(cleaned, format="%a %b %d %H:%M:%S %Y", errors="coerce")

    out = pd.DataFrame({
        "post_id":    "tw_" + df["id"].astype(str),
        "source":     "twitter",
        "text":       df["text"].astype(str),
        "created_at": created,
        "raw_label":  df["target"],
    })
    out["topic"] = out["text"].apply(tag_topic)
    return out

# ------------------------------------------------------------------ DB LOAD -
def load_into_postgres(df: pd.DataFrame) -> None:
    load_dotenv()
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
    )
    cols = ["post_id", "source", "topic", "text", "created_at", "raw_label"]
    records = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df[cols].itertuples(index=False, name=None)
    ]
    with conn, conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE posts;")   # re-running reloads cleanly
        execute_values(
            cur,
            "INSERT INTO posts (post_id, source, topic, text, created_at, raw_label) "
            "VALUES %s ON CONFLICT (post_id) DO NOTHING",
            records, page_size=5_000,
        )
    conn.close()

# --------------------------------------------------------------------- MAIN -
def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.concat([
        load_reddit("reddit_tech.csv"),
        load_reddit("reddit_wsb.csv"),
        load_sentiment140(),
    ], ignore_index=True)

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""]                                   # text is NOT NULL
    df = df.drop_duplicates(subset="post_id", keep="first").reset_index(drop=True)
    df["raw_label"] = df["raw_label"].apply(lambda x: int(x) if pd.notna(x) else None)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df):,} rows -> {OUTPUT_CSV}\n")

    load_into_postgres(df)

    print("Rows by source:\n", df["source"].value_counts(), "\n", sep="")
    print("Rows by topic:\n", df["topic"].value_counts(), "\n", sep="")
    nulls = df["created_at"].isna().groupby(df["source"]).sum()
    print("created_at NULLs by source (high reddit => flip REDDIT_DATE_COL):\n",
          nulls, "\n", sep="")
    print("Sample:\n", df.head(3).to_string(), sep="")

if __name__ == "__main__":
    main()