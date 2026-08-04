import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import kagglehub
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from config.settings import MODEL_CONFIG

INPUT_PATH = "data/processed/call_transcripts.csv"
OUTPUT_PATH = "data/processed/intents_from_audio.csv"
OUTPUT_DIR = Path("models/fine_tuned_intent_distilbert")

INTENT_KEYWORDS = {
    "billing_issue": ["bill", "charge", "payment", "invoice"],
    "account_access": ["login", "password", "sign in", "account access"],
    "technical_issue": ["not working", "error", "crash", "bug", "issue"],
    "cancellation": ["cancel", "unsubscribe", "terminate"],
    "refund_request": ["refund", "money back", "return"],
    "complaint": ["complain", "terrible", "worst", "disappointed"],
}


def label_intent(text: str) -> str:
    """Assigns an intent label to a given text based on keywords."""
    text_l = str(text).lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(word in text_l for word in keywords):
            return intent
    return "other"


def build_intent_labels_from_transcripts():
    """Generates intent labels from local call transcripts if they exist."""
    if not Path(INPUT_PATH).exists():
        print(f"Info: Input transcript file not found at '{INPUT_PATH}'. Skipping local intent labeling.")
        # Create an empty file with headers so the rest of the script doesn't fail
        pd.DataFrame(columns=["text", "intent"]).to_csv(OUTPUT_PATH, index=False)
        return

    print(f"Loading transcripts from '{INPUT_PATH}' to generate local intent labels...")
    df = pd.read_csv(INPUT_PATH)
    if "text" not in df.columns:
        print(f"Warning: 'text' column not found in '{INPUT_PATH}'. Skipping local intent labeling.")
        pd.DataFrame(columns=["text", "intent"]).to_csv(OUTPUT_PATH, index=False)
        return

    df["intent"] = df["text"].apply(label_intent)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Local intent labels saved to '{OUTPUT_PATH}'.")


def load_data(kaggle_path: str, local_path: str):
    """Loads and combines data from the Kaggle dataset and local files."""
    # Load and label Kaggle data
    print(f"Loading and labeling Kaggle data from '{kaggle_path}'...")
    df_kaggle = pd.read_csv(kaggle_path)
    df_kaggle["intent"] = df_kaggle["text"].apply(label_intent)
    df_kaggle = df_kaggle[["text", "intent"]].dropna()

    # Load local data if it exists and is not empty
    df_local = pd.DataFrame()
    if Path(local_path).exists() and os.path.getsize(local_path) > 0:
        print(f"Loading local data from '{local_path}'...")
        df_local = pd.read_csv(local_path)
        df_local = df_local[["text", "intent"]].dropna()

    # Combine datasets
    if not df_local.empty:
        print("Combining Kaggle and local datasets...")
        combined_df = pd.concat([df_kaggle, df_local], ignore_index=True)
    else:
        print("Info: Local data is empty or not found. Using only Kaggle data.")
        combined_df = df_kaggle

    # Clean and prepare final DataFrame
    combined_df = combined_df[combined_df["text"].apply(lambda x: isinstance(x, str))]
    
    # Stratification requires at least 2 samples per class. Remove intents with only one sample.
    intent_counts = combined_df['intent'].value_counts()
    single_occurrence_intents = intent_counts[intent_counts == 1].index
    if len(single_occurrence_intents) > 0:
        print(f"Warning: The following intents have only one sample and will be removed: {list(single_occurrence_intents)}")
        combined_df = combined_df[~combined_df['intent'].isin(single_occurrence_intents)]

    print("Splitting data into training and validation sets...")
    train_df, val_df = train_test_split(combined_df, test_size=0.1, stratify=combined_df["intent"])
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


def tokenize_function(examples, tokenizer, label2id):
    """Tokenizes text and converts string intent labels to integer IDs."""
    result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    if "intent" in examples:
        result["label"] = [label2id[lbl] for lbl in examples["intent"]]
    return result


def main():
    """Main function to run the fine-tuning process."""
    # 1. Build intent labels from local audio transcripts
    build_intent_labels_from_transcripts()

    # 2. Download Kaggle dataset
    print("Downloading Kaggle dataset...")
    dataset_dir = Path(kagglehub.dataset_download("thoughtvector/customer-support-on-twitter"))
    
    # Locate twcs.csv recursively in dataset_dir
    kaggle_csv_path = None
    for csv_candidate in dataset_dir.rglob("*.csv"):
        if csv_candidate.name.lower() == "twcs.csv":
            kaggle_csv_path = str(csv_candidate)
            break
        elif csv_candidate.name.lower() != "sample.csv" and kaggle_csv_path is None:
            kaggle_csv_path = str(csv_candidate)

    if not kaggle_csv_path:
        raise FileNotFoundError(f"Could not find 'twcs.csv' in dataset directory: {dataset_dir}")
    
    print(f"Found Kaggle CSV dataset at: {kaggle_csv_path}")

    # 3. Load and combine both datasets
    train_ds, val_ds = load_data(kaggle_path=kaggle_csv_path, local_path=OUTPUT_PATH)

    # 4. Prepare model and tokenizer with QLoRA quantization
    label_list = sorted(train_ds.unique("intent"))
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["intent_base_model"])
    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer, label2id), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer, label2id), batched=True)

    # Check target device (CUDA vs MPS vs CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    model_kwargs = {
        "num_labels": len(label_list),
        "id2label": id2label,
        "label2id": label2id,
    }

    if device == "mps":
        print("Configuring QLoRA 4-bit BitsAndBytes quantization for MPS...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    # PyTorch SDPA on MPS does not support dropout > 0 during training.
    # Force eager attention implementation when running on MPS.
    if device == "mps":
        model_kwargs["attn_implementation"] = "eager"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CONFIG["intent_base_model"],
        **model_kwargs,
    )

    if device == "cuda":
        print("Preparing quantized model for k-bit training...")
        base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],
        bias="none",
    )

    model = get_peft_model(base_model, peft_config)
    print("LoRA model parameters:")
    model.print_trainable_parameters()

    # 6. Configure and run the Trainer
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=50,
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Starting QLoRA model training...")
    trainer.train()
    print("Training complete.")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"QLoRA Model and tokenizer saved to '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()