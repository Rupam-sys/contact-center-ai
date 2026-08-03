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

import re
import librosa
from transformers import pipeline

from config.settings import ASSEMBLYAI_API_KEY

aai.settings.api_key = ASSEMBLYAI_API_KEY

_local_asr_pipeline = None


def _get_local_asr():
    global _local_asr_pipeline
    if _local_asr_pipeline is None:
        _local_asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return _local_asr_pipeline


def _transcribe_locally(audio_path: str) -> Tuple[str, List[Dict]]:
    """Transcribes audio locally using Hugging Face Whisper without requiring external API keys."""
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
        asr = _get_local_asr()
        res = asr({"raw": speech, "sampling_rate": 16000}, return_timestamps=True)
        transcript = res.get("text", "").strip()
    except Exception as e:
        print(f"Error reading or transcribing audio file locally: {e}")
        transcript = "No audio transcript could be processed."

    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', transcript) if s.strip()]
    if not raw_sentences:
        raw_sentences = [transcript] if transcript else ["No audio content detected."]

    turns = []
    speakers = ["speaker_A", "speaker_B"]
    for idx, sentence in enumerate(raw_sentences):
        turns.append({
            "speaker": speakers[idx % 2],
            "text": sentence
        })
    return transcript, turns


def get_kaggle_audio_files() -> List[str]:
    dataset_path = kagglehub.dataset_download("axondata/call-center-speech-dataset")
    audio_files: List[str] = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".wav", ".mp3")):
                audio_files.append(os.path.join(root, f))
    return audio_files


def transcribe_call(audio_path: str) -> Tuple[str, List[Dict]]:
    api_key = ASSEMBLYAI_API_KEY or os.getenv("ASSEMBLYAI_API_KEY")
    
    if api_key and api_key.strip():
        try:
            aai.settings.api_key = api_key
            config = aai.TranscriptionConfig(speaker_labels=True)
            transcriber = aai.Transcriber()
            transcript_obj = transcriber.transcribe(audio_path, config=config)

            transcript = transcript_obj.text
            turns: List[Dict] = []
            if transcript_obj.utterances:
                for utt in transcript_obj.utterances:
                    turns.append({
                        "speaker": f"speaker_{utt.speaker}",
                        "text": utt.text,
                    })
            else:
                turns.append({"speaker": "speaker_A", "text": transcript or ""})

            return transcript, turns
        except Exception as e:
            print(f"AssemblyAI transcription failed: {e}. Falling back to local Whisper model.")

    # Fallback: transcribe actual audio locally using Hugging Face Whisper
    print("Transcribing audio file locally using Hugging Face Whisper model...")
    return _transcribe_locally(audio_path)


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

