"""
dashboard.py
Streamlit dashboard over the posts + predictions join produced by
export_snapshot.py. Prefers data/processed/dashboard.parquet when present
and falls back to querying Postgres directly -- Week 4 deploys to Streamlit
Community Cloud, which has no database, so the parquet snapshot is committed
to the repo alongside the app.

Usage:
    streamlit run src/dashboard.py
"""

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from wordcloud import WordCloud, STOPWORDS

# --------------------------------------------------------------------- CONFIG
load_dotenv()

PARQUET_PATH = Path("data/processed/dashboard.parquet")
LIVE_HN_DIR = Path("data/live/hn")

DB = dict(
    dbname=os.environ.get("DB_NAME"),
    user=os.environ.get("DB_USER"),
    password=os.environ.get("DB_PASSWORD"),
    host=os.environ.get("DB_HOST"),
    port=os.environ.get("DB_PORT"),
)

QUERY = """
    SELECT p.post_id, p.source, p.topic, p.text, p.created_at,
           pr.model_name, pr.label, pr.score
    FROM posts p
    JOIN predictions pr ON pr.post_id = p.post_id
"""

# Keep in sync with classify_sentiment.MODEL_NAME / finetune.FINETUNED_MODEL_NAME.
# Not imported directly to avoid pulling torch/transformers into the dashboard
# process -- predictions are already computed by the time this app runs.
MODEL_CHOICES = {
    "Zero-shot (3-class) — 81.27% strict, 73.9% coverage":
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "Fine-tuned (2-class) — 86.77% on held-out test set":
        "roberta-sentiment140-finetuned",
}

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#d03b3b",
    "neutral": "#898781",
    "positive": "#0ca30c",
}
SENTIMENT_DASH = {
    "negative": "dot",
    "neutral": "dash",
    "positive": "solid",
}

WORDCLOUD_SAMPLE_SIZE = 5_000
SAMPLE_TABLE_SIZE = 200

# Generic English stopwords miss short/social-media forms that otherwise
# dominate the word cloud, plus leftover HTML-entity fragments ("quot",
# "amp") from posts that predate the entity-unescaping fix in load_data.py.
CUSTOM_STOPWORDS = {
    "will", "now", "day", "one", "go", "got", "know", "think", "u", "im",
    "quot", "amp", "lt", "gt", "ur", "youre", "dont", "didnt", "cant",
    "ill", "ive", "thats", "gonna", "gotta", "rt", "via",
}
WORDCLOUD_STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)

# --------------------------------------------------------------------- DATA
@st.cache_data
def load_data() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
    else:
        try:
            import psycopg2
        except ImportError as e:
            raise RuntimeError(
                f"{PARQUET_PATH} is missing and psycopg2 isn't installed. "
                "Run src/export_snapshot.py to generate the parquet snapshot, "
                "or install psycopg2-binary to query Postgres directly."
            ) from e
        conn = psycopg2.connect(**DB)
        df = pd.read_sql(QUERY, conn)
        conn.close()
    df["is_live"] = False

    live_files = sorted(LIVE_HN_DIR.glob("*.parquet"))
    if live_files:
        live_df = pd.concat([pd.read_parquet(f) for f in live_files], ignore_index=True)
        live_df["is_live"] = True
        df = pd.concat([df, live_df], ignore_index=True)

    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def sentiment_order_present(labels) -> list:
    present = set(labels)
    return [s for s in SENTIMENT_ORDER if s in present]

# --------------------------------------------------------------------- APP
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("Sentiment Analysis Dashboard")

df = load_data()

# ---- sidebar ----
st.sidebar.header("Filters")

model_label = st.sidebar.selectbox("Model", list(MODEL_CHOICES.keys()), index=0)
model_name = MODEL_CHOICES[model_label]
st.sidebar.caption(
    "Live Hacker News rows are only scored by the zero-shot model — "
    "switching to the fine-tuned model hides them from every chart."
)
mdf = df[df["model_name"] == model_name]

sources = sorted(df["source"].unique())
selected_sources = st.sidebar.multiselect("Source", sources, default=sources)

topics = sorted(df["topic"].unique())
selected_topics = st.sidebar.multiselect("Topic", topics, default=topics)

min_date = df["created_at"].min().date()
max_date = df["created_at"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date,
)

min_confidence = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.01)

# ---- apply filters (in pandas, on the already-loaded frame) ----
filtered = mdf[
    mdf["source"].isin(selected_sources)
    & mdf["topic"].isin(selected_topics)
    & (mdf["score"] >= min_confidence)
]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["created_at"].dt.date >= start)
        & (filtered["created_at"].dt.date <= end)
    ]

if filtered.empty:
    st.warning("No posts match the current filters. Try widening the date "
               "range, sources, topics, or lowering the confidence threshold.")
    st.stop()

present_labels = sentiment_order_present(filtered["label"])

# ---- KPI row ----
kpi_cols = st.columns(1 + len(present_labels))
kpi_cols[0].metric("Posts shown", f"{len(filtered):,}")
for col, label in zip(kpi_cols[1:], present_labels):
    pct = (filtered["label"] == label).mean() * 100
    col.metric(label.capitalize(), f"{pct:.1f}%")

# ---- sentiment distribution ----
st.subheader("Sentiment distribution")
dist = (
    filtered["label"].value_counts()
    .reindex(present_labels)
    .rename_axis("label")
    .reset_index(name="count")
)
dist["pct"] = dist["count"] / dist["count"].sum() * 100
fig_dist = px.bar(
    dist, x="label", y="count", color="label",
    category_orders={"label": present_labels},
    color_discrete_map=SENTIMENT_COLORS,
    text=dist["pct"].map(lambda p: f"{p:.1f}%"),
)
fig_dist.update_layout(showlegend=False, xaxis_title=None, yaxis_title="posts")
st.plotly_chart(fig_dist, width="stretch")

# ---- sentiment over time (separate chart per source) ----
st.subheader("Sentiment over time")
st.caption("Twitter and reddit cover very different date ranges and volumes "
           "(twitter: ~2.5 months of 2009 at ~650 posts/day; reddit: ~2.2 "
           "years of 2020-2022 at ~68 posts/day), so each gets its own chart "
           "and its own time bucket.")

TIME_FREQ = {"twitter": ("D", "day"), "reddit": ("W", "week"), "hackernews": ("D", "day")}
reddit_log_scale = False
if "reddit" in selected_sources:
    reddit_log_scale = st.checkbox(
        "Log scale for reddit chart (the Jan 2021 GME spike flattens everything else)",
        value=False,
    )

for source in [s for s in ["twitter", "reddit", "hackernews"] if s in selected_sources]:
    sdf = filtered[filtered["source"] == source]
    if sdf.empty:
        continue
    freq, freq_label = TIME_FREQ[source]
    ts = (
        sdf.groupby([pd.Grouper(key="created_at", freq=freq), "label"])
        .size()
        .rename("count")
        .reset_index()
    )
    source_labels = sentiment_order_present(sdf["label"])
    fig_ts = px.line(
        ts, x="created_at", y="count", color="label",
        category_orders={"label": source_labels},
        color_discrete_map=SENTIMENT_COLORS,
        line_dash="label",
        line_dash_map=SENTIMENT_DASH,
        markers=True,
        title=f"{source.capitalize()} — posts per {freq_label}",
    )
    fig_ts.update_layout(xaxis_title=None, yaxis_title="posts")
    if source == "reddit" and reddit_log_scale:
        fig_ts.update_yaxes(type="log")
    st.plotly_chart(fig_ts, width="stretch")

# ---- word cloud ----
st.subheader("Word cloud")
split_by_sentiment = st.checkbox("Split by sentiment", value=False)


def render_wordcloud(texts: pd.Series):
    if len(texts) > WORDCLOUD_SAMPLE_SIZE:
        texts = texts.sample(WORDCLOUD_SAMPLE_SIZE, random_state=42)
    blob = " ".join(texts.astype(str))
    if not blob.strip():
        st.caption("No text to render.")
        return
    wc = WordCloud(width=800, height=400, background_color=None, mode="RGBA",
                    stopwords=WORDCLOUD_STOPWORDS, collocations=False).generate(blob)
    st.image(wc.to_array(), width="stretch")


if split_by_sentiment:
    wc_cols = st.columns(len(present_labels))
    for col, label in zip(wc_cols, present_labels):
        with col:
            st.caption(label.capitalize())
            render_wordcloud(filtered.loc[filtered["label"] == label, "text"])
else:
    render_wordcloud(filtered["text"])

# ---- sample posts table ----
st.subheader("Sample posts")

if "sample_seed" not in st.session_state:
    st.session_state.sample_seed = 0
if st.button("🔀 Reshuffle sample"):
    st.session_state.sample_seed += 1

sample_n = min(len(filtered), SAMPLE_TABLE_SIZE)
sample_df = filtered.sample(n=sample_n, random_state=st.session_state.sample_seed)
st.caption(f"Random sample of {sample_n:,} (of {len(filtered):,} filtered posts).")
st.dataframe(
    sample_df[["source", "text", "label", "score"]].rename(
        columns={"source": "Source", "text": "Text",
                 "label": "Predicted label", "score": "Confidence"}
    ),
    width="stretch",
    height=400,
)
