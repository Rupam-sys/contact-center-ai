import streamlit as st
import os
import kagglehub
from pathlib import Path
from src.pipeline import run_pipeline

st.set_page_config(layout="wide")

st.title("Contact Center AI – Call Analytics Demo")

st.markdown(
    """
    Upload a call recording (or use the demo file) and enter the customer's name. 
    Then, run the pipeline to see the generated analysis, including a transcript, 
    sentiment analysis, intent classification, a CRM-style summary, and a draft follow-up email.
    """
)

def display_results(result):
    """Displays the pipeline results in the Streamlit app."""
    st.subheader("Transcript")
    st.write(result.get("transcript", "No transcript available."))

    st.subheader("Turn-by-turn Sentiment")
    if result.get("sentiment"):
        for turn in result["sentiment"]:
            st.write(f"{turn.get('speaker', 'Unknown').capitalize()}: {turn.get('text', '')}  ➜  {turn.get('sentiment', 'N/A')}")
    else:
        st.write("No sentiment analysis available.")

    st.subheader("Intent & Compliance")
    if result.get("intent"):
        st.write(f"Detected intent: {result['intent'].get('intent', 'N/A')}")
        st.json(result['intent'].get("compliance", {}))
    else:
        st.write("No intent or compliance analysis available.")

    st.subheader("CRM Summary")
    st.write(result.get("summary", "No summary available."))

    st.subheader("Follow-up Email Draft")
    st.code(result.get("email", "No email draft available."), language="text")

@st.cache_data
def get_demo_audio_path():
    """Downloads the dataset and returns the path to a demo audio file."""
    try:
        st.info("Downloading demo audio dataset from Kaggle Hub...")
        dataset_path = Path(kagglehub.dataset_download("axondata/call-center-speech-dataset"))
        # Find the first .wav file in the dataset
        audio_files = list(dataset_path.glob("**/*.wav"))
        if not audio_files:
            st.error("No .wav files found in the downloaded dataset.")
            return None
        st.success("Demo audio downloaded successfully.")
        return str(audio_files[0])
    except Exception as e:
        st.error(f"Failed to download or access the dataset: {e}")
        return None

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Inputs")
    customer_name = st.text_input("Customer Name", "Valued Customer")
    audio_file = st.file_uploader("Upload call audio (WAV/MP3)", type=["wav", "mp3"])
    
    run_demo_button = st.button("Run Demo Pipeline")
    run_upload_button = st.button("Run Pipeline on Uploaded Audio", disabled=not audio_file)

# --- Main Content Area ---
if run_demo_button:
    demo_audio_path = get_demo_audio_path()
    if demo_audio_path:
        with st.spinner("Running demo pipeline..."):
            st.audio(demo_audio_path)
            result = run_pipeline(demo_audio_path, customer_name)
            st.success("Demo pipeline executed successfully!")
            display_results(result)

elif run_upload_button and audio_file is not None:
    with st.spinner("Processing uploaded audio..."):
        # Save the uploaded file to a temporary location
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        uploaded_audio_path = os.path.join(temp_dir, audio_file.name)
        with open(uploaded_audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        st.audio(uploaded_audio_path)
        
        result = run_pipeline(uploaded_audio_path, customer_name)
        st.success("Pipeline executed on uploaded audio successfully!")
        display_results(result)
        
        # Clean up the temporary file
        os.remove(uploaded_audio_path)

else:
    st.info("Please upload an audio file or run the demo.")
