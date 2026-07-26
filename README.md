# Sentiment Analysis Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alaraj-sentiment-analysis-dashboard.streamlit.app)

**Live app: https://alaraj-sentiment-analysis-dashboard.streamlit.app**

An ETL + ML pipeline that normalizes Reddit posts (r/technology, r/wallstreetbets)
and Twitter posts (the sentiment140 dataset) into a shared Postgres schema,
classifies each post's sentiment with a transformer model, and exposes the
result through a Streamlit dashboard. Two models are compared: the pretrained
`cardiffnlp/twitter-roberta-base-sentiment-latest` (3-class, zero-shot) and a
version of the same checkpoint fine-tuned into a 2-class head on sentiment140's
labels. Built with Python, pandas, psycopg2, PyTorch/Transformers, and Streamlit.

## Architecture

Two independent tracks meet only when the dashboard loads.

### Historical track (frozen)

`load_data.py` loads three Kaggle datasets — Reddit r/technology, Reddit
r/wallstreetbets, and the sentiment140 Twitter set — into a local Postgres
`posts` table: 103,992 posts. `classify_sentiment.py` scores them into a
`predictions` table. `export_snapshot.py` joins the two into
`data/processed/dashboard.parquet`, the file the deployed dashboard actually
reads. This file is frozen: it changes only when someone reruns
`export_snapshot.py` by hand, after a manual reload or reclassification.

### Live track (append-only)

A GitHub Actions workflow (`.github/workflows/ingest-hn.yml`, see Live
ingestion below) runs `ingest_hn.py --sink parquet` on a schedule. Each run
writes to `data/live/hn/<UTC date>.parquet` — one file per day. It never
writes to `dashboard.parquet` and never touches Postgres.

### Why two tracks instead of one growing file

CI never writes to the file the dashboard reads for its historical data, so
a bug in the live pipeline can't corrupt the archive, and there's no write
race between a scheduled Action and someone running `export_snapshot.py` by
hand at the same time. The archive also stays reproducible from Postgres —
regenerating it is always "rerun `export_snapshot.py`," never "reconcile it
against whatever the live track wrote since."

Byte growth is a secondary factor, and a smaller one than it first looks.
Rewriting `dashboard.parquet` (currently ~14MB) twice daily sounds
expensive, but git delta-compresses repeated rewrites of a mostly-unchanged
binary file well: simulating 10 twice-daily rewrites, each appending ~800
rows, grew packed git history by roughly 70-100MB/year extrapolated — not
the multi-gigabyte figure a naive "new full copy every commit" estimate
would suggest. That number is a floor, not a realistic estimate: the
simulation only appended to an otherwise-untouched file, so unchanged row
groups compressed away almost for free. A real twice-daily regeneration
re-runs the whole Postgres join and re-encodes the entire file from
scratch, shifting row-group boundaries and dictionary encoding throughout —
that delta-compresses far worse than the simulation. Writing a separate,
untouched file per day sidesteps the question: each partition is written
once and never rewritten, and measured growth is linear — roughly
35-42MB/year at the current ~700-900 unique posts/day rate.

### Divergence is by design

The GitHub Actions runner has no network path to a local Postgres instance,
so Postgres and the live parquet partitions are independent by construction
and will drift apart over time. Postgres is not the source of truth for
Hacker News data — `data/live/hn/*.parquet` is.

### Loading

`dashboard.py`'s `load_data()` reads the archive (parquet, or Postgres as a
fallback), then concatenates every `data/live/hn/*.parquet` partition it
finds on top. Rows get an `is_live` column so the two tracks stay
distinguishable after the concat. A missing or empty `data/live/hn/`
directory — the normal state right after cloning, before any Action has run
— is a no-op, not an error.

## Live ingestion

`ingest_hn.py` pulls current stories from the Hacker News Firebase API
(`https://hacker-news.firebaseio.com/v0/`) — no authentication required.
`.github/workflows/ingest-hn.yml` runs it twice a day, at 03:07 and 15:07
UTC, and commits the result back to `main` if anything changed.

### Coverage arithmetic

`newstories.json` returns at most 500 story IDs, spanning roughly the last
18 hours of submissions. Hacker News produces roughly 670 stories/day. A
single run per day would therefore silently miss around 26% of stories —
the ones that scroll off the end of that 500-ID window before the next run
picks it up.

Running twice a day instead of once gives two overlapping ~18-hour windows.
The two runs are not additive, though — most of the second run's 500 IDs
were already seen by the first. The same-day merge and two-prior-day dedup
in `write_parquet_sink` (`ingest_hn.py`) collapse that overlap. In practice,
expect roughly 700-900 unique posts/day, not a flat 1,000+ (500 IDs × 2
runs).

### Why Hacker News

The original plan was to add a second live source from Reddit. That API
access request was refused. Hacker News's Firebase API needs no
registration, no key, and no approval — which is why it replaced Reddit as
the live-ingestion source.

## Findings

Evaluation results so far. Ground truth is sentiment140's emoticon-derived
labels (0=negative, 4=positive) on twitter rows only; reddit has no
ground-truth labels.

### Zero-shot baseline

`cardiffnlp/twitter-roberta-base-sentiment-latest`, unmodified, is a 3-class
model (negative/neutral/positive) scored against sentiment140's 2-class
ground truth. Two different scoring populations, reported separately:

- **Full labeled set** (~50,000 twitter rows): 81.65% strict accuracy at
  73.62% coverage; 77.40% forced accuracy at 100% coverage.
- **Same 10,000-row held-out `test_split` used to evaluate the fine-tuned
  model below**: 81.27% strict accuracy at 73.93% coverage; 77.17% forced
  accuracy at 100% coverage.

Both populations are below an 85% target. The two are close (as expected —
`test_split` is a random subset of the full set), but only the second
figure is comparable to the fine-tuned model's score, below.

### Fine-tuned model

The 3-class head was replaced with a 2-class (negative/positive) head and
fine-tuned for 1 epoch on 20,000 sentiment140 rows (10,000/class). It scores
**86.77%** accuracy on the same 10,000-row held-out `test_split` (5,000/class)
used for the zero-shot comparison above.

The test post_ids were written to the `test_split` table before training
started and never regenerated; a direct check confirmed zero overlap
between the training set and `test_split`.

**86.77% (fine-tuned) vs. 81.27% (zero-shot strict) is the valid head-to-head
comparison** — both are scored on the identical 10,000 rows. The 81.65%
full-set figure above is not comparable to 86.77%; it's a different, larger
population that overlaps with the fine-tuned model's training data.

86.77% applies only to the 10,000-row held-out split — 20,000 of the
~50,000 twitter rows in the corpus were used as training data for this
model, so the same accuracy figure does not describe the full displayed
corpus.

### Data cleaning: HTML entities

3,038 posts (2.92% of the corpus) carried un-unescaped HTML entities
(`&quot;`, `&amp;`, `&lt;`, `&gt;`) left over from the source CSVs. These
were cleaned in place and re-classified by both models; `load_data.py` now
unescapes on load so future imports are unaffected. The fine-tuned accuracy
moved by exactly one row (86.76% → 86.77%) — the cleanup had no material
effect on model performance.

### Zero-shot vs. fine-tuned agreement

Restricting to rows where the zero-shot model committed to negative or
positive (excluding its neutral calls), the two models agree on polarity:

- reddit: 86.60%
- twitter: 85.02%

This is an agreement rate between two models, not an accuracy figure —
reddit has no ground-truth labels, so there is no way to know which model
(if either) is correct on reddit rows.

### Calibration caveat: fine-tuned model on reddit

The fine-tuned model is a 2-class head and cannot output neutral. On reddit
text with no sentiment content — questions, logistics posts — it still
assigns a confident negative or positive label, sometimes above 0.9:

- "How do i invest in GameStop? I don't know anything about stock markets"
  → negative (0.979)
- "HOW CAN I BUY STOCK????!!!" → negative (0.928)

This follows directly from training data: sentiment140's labels are derived
from emoticons on a forced binary (negative/positive) scale, so the model
never saw a "no sentiment" class during training and has no way to express
one at inference time.

### Why the dashboard defaults to zero-shot

Despite the fine-tuned model's higher accuracy on the held-out test set,
the dashboard's default model is the zero-shot checkpoint. Reddit is the
majority of the displayed corpus and has no ground-truth labels; the
zero-shot model's ability to abstain (predict neutral) is preferred there
over the fine-tuned model's forced, sometimes falsely confident,
negative/positive call on sentiment-free text.

### Live data is zero-shot only

`data/live/hn/*.parquet` rows carry only zero-shot
(`cardiffnlp/twitter-roberta-base-sentiment-latest`) predictions. The
fine-tuned model is ~500MB and gitignored, so the CI runner has no way to
load it without committing a 500MB binary to the repo or adding a
model-download step — it does not run against Hacker News at all.
Selecting the fine-tuned model in the dashboard's model selector hides
every Hacker News row from every chart.

### Hacker News skews heavily neutral

Classifying the first 500 Hacker News stories with the zero-shot model gave
397 neutral / 74 negative / 29 positive — about 79% neutral. Hacker News
headlines are mostly descriptive ("X releases Y", "Show HN: Z") rather than
opinionated, and the model reads that correctly as unemotional. Including
Hacker News visibly shifts the dashboard's overall sentiment distribution
toward neutral. This is a property of the source, not a classifier fault.

### Three disjoint time ranges

The archive spans two eras that don't overlap: sentiment140/Twitter is
~2.5 months of 2009, Reddit is ~2.2 years of 2020-2022. Hacker News is
present-day. There's no shared timeline across all three sources, which is
why "Sentiment over time" renders one chart per source instead of a single
combined chart.

## Running it

Python 3.12 is what CI and Streamlit Community Cloud actually run. Local
development on this repo has been on Python 3.14; both work, but 3.12 is
the version anything deployed runs on.

### Requirements files

Three requirements files exist, and they are not interchangeable:

- `requirements.txt` — what Streamlit Community Cloud installs from.
  Runtime only: no `torch`/`transformers`. The dashboard never loads a
  model itself; it only reads predictions already written to parquet or
  Postgres.
- `requirements-dev.txt` — full local set: `torch`, `transformers`,
  `psycopg2-binary`. Needed to load, classify, fine-tune, or evaluate.
- `requirements-action.txt` — the GitHub Actions set, for `ingest_hn.py
  --sink parquet`. `torch` installs separately, from the CPU-only wheel
  index (`--index-url https://download.pytorch.org/whl/cpu`) — the default
  PyPI wheel pulls in ~3GB of CUDA a GitHub-hosted runner can't use.

### Fastest path: dashboard only

```bash
git clone <repo>
pip install -r requirements.txt
streamlit run src/dashboard.py
```

No database, no model download. `dashboard.py` reads the committed parquet
files (`data/processed/dashboard.parquet` plus any `data/live/hn/*.parquet`
partitions) directly. This is the fastest way to see the app running.

### Full local pipeline

Needs Postgres running locally and `pip install -r requirements-dev.txt`.

1. Start Postgres. On WSL it does not start automatically —
   `sudo service postgresql start` (or the equivalent for your setup).
2. Create a `.env` (gitignored) with `DB_NAME`, `DB_USER`, `DB_PASSWORD`,
   `DB_HOST`, `DB_PORT`.
3. `python src/load_data.py` — builds `data/processed/posts_unified.csv`
   and loads the `posts` table (`TRUNCATE` + insert; idempotent, not
   additive).
4. `python src/classify_sentiment.py` — scores `posts` with the zero-shot
   model into `predictions`. `--validate` classifies 20 rows and prints,
   no DB write; `--limit N` classifies only the first N unclassified rows.
5. `python src/finetune.py` — fine-tunes the 2-class head on sentiment140.
   No arguments; the held-out test split is fixed on first run and reused
   on every run after.
6. `python src/evaluate.py` — scores a model against the held-out test
   set. Default is the zero-shot model; `--model finetuned` scores the
   fine-tuned one instead.
7. `python src/export_snapshot.py` — writes
   `data/processed/dashboard.parquet` from the current `posts` and
   `predictions` tables.
8. `streamlit run src/dashboard.py`.

### Live ingestion, run locally

`ingest_hn.py` has two sinks:

- `--sink postgres` (default) — inserts into the local `posts` table, same
  as reddit/twitter rows. Requires the same `.env`/Postgres setup as the
  full pipeline above.
- `--sink parquet` — what the GitHub Action runs. Classifies with the
  zero-shot model and writes/merges `data/live/hn/<UTC date>.parquet`. Needs
  no Postgres connection at all.

`--dry-run` fetches and builds rows without writing to either sink — useful
for checking what a run would do first.

## Deployment

Live at **https://alaraj-sentiment-analysis-dashboard.streamlit.app**, deployed
on Streamlit Community Cloud from this repo's `main` branch (`src/dashboard.py`
as the entrypoint).

Two requirements files:

- `requirements.txt` — only what `src/dashboard.py` imports at runtime. No
  `torch`/`transformers`: the dashboard reads predictions already written to
  `data/processed/dashboard.parquet` (or Postgres) and never loads a model
  itself. **This is the file Streamlit Community Cloud installs from** — it
  reads a file named exactly `requirements.txt` at the repo root, with no way
  to point it at an alternate filename, so the runtime-only set has to live
  at that path.
- `requirements-dev.txt` — full dev set (load/train/classify/evaluate +
  dashboard). Includes `torch`, `transformers`, and `psycopg2-binary` for
  running the models locally: `pip install -r requirements-dev.txt`.

Running `pip install -r requirements.txt` alone will not give you `torch` —
use `requirements-dev.txt` for local model work.

`data/processed/dashboard.parquet` is committed (see `.gitignore`) so the
deployed app has data to read without a database — Streamlit Community Cloud
has no Postgres of its own.
