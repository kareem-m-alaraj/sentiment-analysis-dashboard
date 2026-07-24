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
