import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Tuple
import os
import kagglehub
import assemblyai as aai
import pandas as pd

from config.settings import ASSEMBLYAI_API_KEY

aai.settings.api_key = ASSEMBLYAI_API_KEY


def get_kaggle_audio_files() -> List[str]:
    dataset_path = kagglehub.dataset_download("axondata/call-center-speech-dataset")
    audio_files: List[str] = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".wav", ".mp3")):
                audio_files.append(os.path.join(root, f))
    return audio_files


def transcribe_call(audio_path: str) -> Tuple[str, List[Dict]]:
    config = aai.TranscriptionConfig(
        speaker_labels=True,
    )
    transcriber = aai.Transcriber()
    transcript_obj = transcriber.transcribe(audio_path, config=config)

    transcript = transcript_obj.text
    turns: List[Dict] = []
    for utt in transcript_obj.utterances:
        turns.append({
            "speaker": f"speaker_{utt.speaker}",
            "text": utt.text,
        })

    return transcript, turns


def build_transcript_corpus(max_calls: int = 200, output_path: str = "data/processed/call_transcripts.csv"):
    """
    Transcribe a subset of Kaggle audio files and save a CSV:
    columns: call_id, text
    You will later add 'intent' labels to this CSV.
    """
    audio_files = get_kaggle_audio_files()
    rows = []

    for i, audio_path in enumerate(audio_files[:max_calls]):
        transcript, turns = transcribe_call(audio_path)
        rows.append({
            "call_id": os.path.basename(audio_path),
            "text": transcript,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def main():
    print("Building transcript corpus...")
    build_transcript_corpus()
    print("Transcript corpus built successfully.")


if __name__ == "__main__":
    main()

