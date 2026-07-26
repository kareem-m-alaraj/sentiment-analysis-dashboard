"""
topics.py — keyword-based topic tagging shared by load_data.py and ingest_hn.py.

Dependency-free (stdlib only) so ingestion scripts can import it without
pulling in psycopg2/pandas.
"""

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
