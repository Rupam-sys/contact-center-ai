import pytest
from src.pipeline import run_pipeline

# Mock data to be returned by the patched functions
MOCK_TRANSCRIPT = "Hello, this is a test."
MOCK_TURNS = [{"speaker": "SPEAKER_00", "text": "Hello, this is a test."}]
MOCK_SENTIMENT = [{"speaker": "SPEAKER_00", "text": "Hello, this is a test.", "sentiment": "neutral"}]
MOCK_INTENT = {"intent": "test_intent", "compliance": {"contains_pii": False}}
MOCK_SUMMARY = "This is a test summary."
MOCK_EMAIL = "This is a test email."

@pytest.fixture
def mock_pipeline_dependencies(monkeypatch):
    """Mocks all external dependencies of the pipeline for isolated testing."""
    monkeypatch.setattr("src.pipeline.transcribe_call", lambda audio_path: (MOCK_TRANSCRIPT, MOCK_TURNS))
    monkeypatch.setattr("src.pipeline.analyze_sentiment_turns", lambda turns: MOCK_SENTIMENT)
    monkeypatch.setattr("src.pipeline.detect_intent_and_compliance", lambda transcript: MOCK_INTENT)
    monkeypatch.setattr("src.pipeline.generate_crm_summary", lambda transcript: MOCK_SUMMARY)
    monkeypatch.setattr("src.pipeline.generate_followup_email", lambda summary, customer_name: MOCK_EMAIL)

def test_run_pipeline_structure_and_content(mock_pipeline_dependencies):
    """
    Tests the pipeline with mocked dependencies to ensure it returns the expected structure and content.
    """
    # Define test inputs
    test_audio_path = "tests/test_audio.wav"
    test_customer_name = "John Doe"

    # Run the pipeline
    result = run_pipeline(test_audio_path, test_customer_name)

    # Assert that the output has the correct structure and keys
    expected_keys = ["customer_name", "transcript", "turns", "sentiment", "intent", "summary", "email"]
    assert all(key in result for key in expected_keys)

    # Assert that the content matches the mock data
    assert result["customer_name"] == test_customer_name
    assert result["transcript"] == MOCK_TRANSCRIPT
    assert result["turns"] == MOCK_TURNS
    assert result["sentiment"] == MOCK_SENTIMENT
    assert result["intent"] == MOCK_INTENT
    assert result["summary"] == MOCK_SUMMARY
    assert result["email"] == MOCK_EMAIL
    assert isinstance(result["summary"], str)
