import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from config.settings import MODEL_CONFIG

_tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["sentiment_model"])
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_CONFIG["sentiment_model"])

LABELS = ["negative", "neutral", "positive"]  # cardiffnlp twitter-roberta mapping


def _predict_sentiment(text: str) -> str:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    scores = outputs.logits[0]
    label_id = int(torch.argmax(scores))
    return LABELS[label_id]


def analyze_sentiment_turns(turns: List[Dict]) -> List[Dict]:
    """
    Add sentiment label to each turn.
    """
    results = []
    for turn in turns:
        sentiment = _predict_sentiment(turn["text"])
        results.append({**turn, "sentiment": sentiment})
    return results