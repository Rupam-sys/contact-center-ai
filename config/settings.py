import os
from pathlib import Path
from dotenv import load_dotenv

# Define base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env in project root
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

MODEL_CONFIG = {
    "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment",
    "intent_base_model": "distilbert/distilbert-base-uncased",
    "summarization_model": "meta-llama/Llama-3.1-8B-Instruct",
    "email_model": "meta-llama/Llama-3.1-8B-Instruct"
}
