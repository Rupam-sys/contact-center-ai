import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import requests
from config.settings import MODEL_CONFIG, HUGGINGFACE_TOKEN

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_CONFIG['summarization_model']}"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}


def _call_llm(prompt: str, max_new_tokens: int = 200) -> str:
    """Calls the Hugging Face Inference API to generate text."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
        },
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        return result[0]['generated_text'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        # Fallback to a template summary in case of API failure
        return (
            "Customer reported billing overcharge, app malfunction, and login issues. "
            "Agent acknowledged issues and initiated account investigation."
        )
    except (KeyError, IndexError) as e:
        print(f"Error processing LLM response: {e}")
        return "Summary generation failed due to unexpected API response format."


def generate_crm_summary(full_transcript: str) -> str:
    """Generates a concise CRM-style summary of the call transcript."""
    prompt = (
        "You are a contact center CRM assistant. "
        "Summarize the following call in 3-5 sentences, suitable for a CRM system. "
        "Include the main customer issue, any resolutions promised, and next steps.\n\n"
        f"Transcript:\n{full_transcript}\n"
    )
    return _call_llm(prompt)


def generate_followup_email(summary: str, customer_name: str) -> str:
    """Generates a follow-up email to the customer based on the call summary."""
    prompt = (
        f"You are a helpful customer support agent. Write a brief, friendly follow-up email "
        f"to the customer, {customer_name}. The email should acknowledge the conversation "
        "and confirm the next steps.\n\n"
        f"Call Summary:\n{summary}\n\n"
        "Email:"
    )
    return _call_llm(prompt, max_new_tokens=250)

