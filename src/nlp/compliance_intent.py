from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

from config.settings import MODEL_CONFIG

INTENT_MODEL_DIR = Path("models/fine_tuned_intent_distilbert")

# Fallback to base model if fine-tuned model has not been trained/saved yet
if INTENT_MODEL_DIR.exists() and (INTENT_MODEL_DIR / "config.json").exists():
    model_path = str(INTENT_MODEL_DIR)
else:
    model_path = MODEL_CONFIG.get("intent_base_model", "distilbert/distilbert-base-uncased")

_intent_tokenizer = AutoTokenizer.from_pretrained(model_path)
_intent_model = AutoModelForSequenceClassification.from_pretrained(model_path)


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