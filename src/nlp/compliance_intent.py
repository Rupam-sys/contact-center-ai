from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

from config.settings import MODEL_CONFIG

INTENT_MODEL_DIR = Path("models/fine_tuned_intent_distilbert")

from peft import PeftModel

DEFAULT_INTENT_LABELS = {
    0: "account_access",
    1: "billing_issue",
    2: "cancellation",
    3: "complaint",
    4: "other",
    5: "refund_request",
    6: "technical_issue",
}

# Fallback logic: check for QLoRA adapter, full fine-tuned model, or base model
if INTENT_MODEL_DIR.exists() and (INTENT_MODEL_DIR / "adapter_config.json").exists():
    _intent_tokenizer = AutoTokenizer.from_pretrained(str(INTENT_MODEL_DIR))
    base_model_name = MODEL_CONFIG.get("intent_base_model", "distilbert/distilbert-base-uncased")
    
    if (INTENT_MODEL_DIR / "config.json").exists():
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(str(INTENT_MODEL_DIR))
        _base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, config=config)
    else:
        num_labels = 7
        safetensors_path = INTENT_MODEL_DIR / "adapter_model.safetensors"
        bin_path = INTENT_MODEL_DIR / "adapter_model.bin"
        if safetensors_path.exists():
            from safetensors.torch import load_file
            weights = load_file(str(safetensors_path))
            for k, v in weights.items():
                if "classifier.weight" in k or "score.weight" in k:
                    num_labels = v.shape[0]
                    break
        elif bin_path.exists():
            weights = torch.load(str(bin_path), map_location="cpu")
            for k, v in weights.items():
                if "classifier.weight" in k or "score.weight" in k:
                    num_labels = v.shape[0]
                    break
        
        kwargs = {"num_labels": num_labels}
        if num_labels == len(DEFAULT_INTENT_LABELS):
            kwargs["id2label"] = DEFAULT_INTENT_LABELS
            kwargs["label2id"] = {v: k for k, v in DEFAULT_INTENT_LABELS.items()}

        _base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, **kwargs)

    _intent_model = PeftModel.from_pretrained(_base_model, str(INTENT_MODEL_DIR))
elif INTENT_MODEL_DIR.exists() and (INTENT_MODEL_DIR / "config.json").exists():
    _intent_tokenizer = AutoTokenizer.from_pretrained(str(INTENT_MODEL_DIR))
    _intent_model = AutoModelForSequenceClassification.from_pretrained(str(INTENT_MODEL_DIR))
else:
    model_path = MODEL_CONFIG.get("intent_base_model", "distilbert/distilbert-base-uncased")
    _intent_tokenizer = AutoTokenizer.from_pretrained(model_path)
    _intent_model = AutoModelForSequenceClassification.from_pretrained(model_path)


def _predict_intent(text: str) -> str:
    if not text or not text.strip():
        return "general_inquiry"
    inputs = _intent_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _intent_model(**inputs)
    logits = outputs.logits[0]
    label_id = int(torch.argmax(logits))
    if hasattr(_intent_model.config, "id2label") and _intent_model.config.id2label:
        label = _intent_model.config.id2label.get(label_id, f"intent_{label_id}")
    else:
        label = f"intent_{label_id}"
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