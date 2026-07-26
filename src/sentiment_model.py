"""
sentiment_model.py — cardiffnlp/twitter-roberta-base-sentiment-latest wrapper.

No DB dependency: classify_texts() takes and returns plain Python, so it can
be imported standalone (e.g. by a workflow classifying a parquet batch with
no Postgres access) as well as by classify_sentiment.py.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# model's own label order for this checkpoint: 0=neg, 1=neu, 2=pos
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
MAX_TOKENS = 128         # tweets/reddit titles are short; 128 is plenty

_tok = None
_model = None

def load_model():
    """Loads and caches the tokenizer/model on first call; a no-op after."""
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tok, _model


@torch.no_grad()
def classify_texts(texts):
    """texts: list[str] -> list[(label, score)], aligned. Loads the model on
    first call if it hasn't been loaded yet."""
    tok, model = load_model()
    enc = tok(texts, padding=True, truncation=True,
              max_length=MAX_TOKENS, return_tensors="pt")
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    conf, idx = torch.max(probs, dim=-1)
    return [(ID2LABEL[i.item()], round(c.item(), 4))
            for i, c in zip(idx, conf)]
