from src.speech.transcription import transcribe_call
from src.nlp.sentiment import analyze_sentiment_turns
from src.nlp.compliance_intent import detect_intent_and_compliance
from src.nlp.summarization_email_generator import generate_crm_summary, generate_followup_email


def run_pipeline(audio_path: str, customer_name: str = "Valued Customer"):
    transcript, turns = transcribe_call(audio_path)
    sentiment_results = analyze_sentiment_turns(turns)
    intent_results = detect_intent_and_compliance(transcript)
    summary = generate_crm_summary(transcript)
    email = generate_followup_email(summary, customer_name)

    return {
        "customer_name": customer_name,
        "transcript": transcript,
        "turns": turns,
        "sentiment": sentiment_results,
        "intent": intent_results,
        "summary": summary,
        "email": email,
    }
