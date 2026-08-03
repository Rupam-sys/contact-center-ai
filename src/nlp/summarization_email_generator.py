import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import re
import requests
from config.settings import MODEL_CONFIG, HUGGINGFACE_TOKEN

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_CONFIG['summarization_model']}"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"} if HUGGINGFACE_TOKEN else {}


def _call_llm(prompt: str, max_new_tokens: int = 200) -> str:
    """Calls the Hugging Face Inference API to generate text if a token is present."""
    if not HUGGINGFACE_TOKEN:
        return None  # Trigger fallback

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
        },
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '').strip()
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text'].strip()
        return None
    except Exception as e:
        print(f"Notice: Hugging Face LLM API call failed ({e}). Using dynamic transcript fallback.")
        return None


def generate_crm_summary(full_transcript: str) -> str:
    """Generates a concise CRM-style summary of the call transcript."""
    prompt = (
        "You are a contact center CRM assistant. "
        "Summarize the following call in 3-5 sentences, suitable for a CRM system. "
        "Include the main customer issue, any resolutions promised, and next steps.\n\n"
        f"Transcript:\n{full_transcript}\n"
    )
    llm_output = _call_llm(prompt)
    if llm_output:
        return llm_output

    # Dynamic fallback based on actual transcript content
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_transcript) if len(s.strip()) > 10]
    if not sentences:
        return "Call recording logged. Customer support agent initiated interaction."

    excerpt = " ".join(sentences[:3])
    return (
        f"Call Summary: Customer interaction logged. Key discussion point: \"{excerpt}\" "
        "Support agent addressed inquiry and confirmed next steps for resolution."
    )


def generate_followup_email(summary: str, customer_name: str) -> str:
    """Generates a follow-up email to the customer based on the call summary."""
    prompt = (
        f"You are a helpful customer support agent. Write a brief, friendly follow-up email "
        f"to the customer, {customer_name}. The email should acknowledge the conversation "
        "and confirm the next steps.\n\n"
        f"Call Summary:\n{summary}\n\n"
        "Email:"
    )
    llm_output = _call_llm(prompt, max_new_tokens=250)
    if llm_output:
        return llm_output

    # Dynamic email template using summary and customer name
    return (
        f"Dear {customer_name},\n\n"
        "Thank you for contacting customer support today.\n\n"
        "Here is a summary of our discussion:\n"
        f"{summary}\n\n"
        "We are actively monitoring your request and will provide additional updates as needed. "
        "Please feel free to reply directly to this email if you have any questions.\n\n"
        "Best regards,\n"
        "Customer Support Team"
    )

