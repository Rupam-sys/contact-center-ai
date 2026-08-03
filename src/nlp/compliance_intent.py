from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

INTENT_MODEL_DIR = Path("models/fine_tuned_intent_distilbert")

_intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_DIR)
_intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_DIR)


def _predict_intent(text: str) -> str:
    inputs = _intent_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = _intent_model(**inputs)
    logits = outputs.logits[0]
    label_id = int(torch.argmax(logits))
    label = _intent_model.config.id2label[label_id]
    return label


def _check_compliance(transcript: str) -> Dict:
    """
    Very simple rule-based compliance checker:
    - Check if greeting exists
    - Check if apology exists when negative words are present
    - Check if closing exists
    """
    lower = transcript.lower()
    compliance_flags = {
        "greeting": any(word in lower for word in ["hello", "hi", "thank you for calling"]),
        "apology": ("sorry" in lower) or ("apologize" in lower),
        "closing": any(word in lower for word in ["thank you", "have a nice day", "goodbye"])
    }

    # Overall compliance satisfies at least greeting and closing
    compliance_flags["overall_compliant"] = compliance_flags["greeting"] and compliance_flags["closing"]
    return compliance_flags


def detect_intent_and_compliance(transcript: str) -> Dict:
    """
    Predict overall intent from full transcript and check compliance.
    """
    intent_label = _predict_intent(transcript)
    compliance = _check_compliance(transcript)
    return {
        "intent": intent_label,
        "compliance": compliance
    }