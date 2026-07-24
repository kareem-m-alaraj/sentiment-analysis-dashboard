"""
export_snapshot.py
Dumps the posts + predictions join that dashboard.py reads to a parquet
snapshot at data/processed/dashboard.parquet. dashboard.py prefers this file
when present and only falls back to Postgres if it's missing -- Week 4 deploys
to HF Spaces, which has no database.

Usage:
    python src/export_snapshot.py
"""

import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

OUTPUT_PARQUET = Path("data/processed/dashboard.parquet")

DB = dict(
    dbname=os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    host=os.environ["DB_HOST"],
    port=os.environ["DB_PORT"],
)

QUERY = """
    SELECT p.post_id, p.source, p.topic, p.text, p.created_at,
           pr.model_name, pr.label, pr.score
    FROM posts p
    JOIN predictions pr ON pr.post_id = p.post_id
"""


def main():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql(QUERY, conn)
    conn.close()

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Wrote {len(df):,} rows -> {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
