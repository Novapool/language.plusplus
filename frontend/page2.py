import streamlit as st
import numpy as np
import tempfile
import os
import base64

from gpt import gpt

from st_audiorec import st_audiorec

import azure.cognitiveservices.speech as speechsdk

# Replace with your own subscription key and service region (e.g., "westus").
SUBSCRIPTION_KEY = st.secrets["SUBSCRIPTION_KEY"]
SERVICE_REGION =  st.secrets["SERVICE_REGION"]

# Set up the speech configuration

def getRating(file_name, lang):

    speech_config = speechsdk.SpeechConfig(subscription=SUBSCRIPTION_KEY, region=SERVICE_REGION)


    # Create a pronunciation assessment configuration
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text="",
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme)

    # Create a recognizer with the given settings
    audio_config = speechsdk.audio.AudioConfig(filename=file_name)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config, language=lang)

    # Apply the pronunciation assessment configuration to the recognizer
    pronunciation_config.apply_to(recognizer)

    print("Processing the WAV file...")

    # Start the recognition
    result = recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
        print("Pronunciation Assessment Result:")
        return (result.text, pronunciation_result.accuracy_score, pronunciation_result.fluency_score)
    else:
        return ("", 0.0, 0.0)

def page2():

    # Custom CSS for styling and hiding Streamlit elements
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #ffffff;
            color: #000000;
        }
        .stApp {
            background-color: #ffffff;
        }
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .content {
            text-align: center;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .title {
            font-size: 5em;
            font-weight: 700;
            color: #6a1b9a;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .subtitle {
            font-size: 2em;
            font-weight: 300;
            color: #9c27b0;
            margin-bottom: 30px;
        }
        .description {
            font-size: 1.2em;
            color: #333333;
            margin-bottom: 40px;
            line-height: 1.6;
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px;
        }
        .stButton button {
            background-color: #6a1b9a;
            color: #ffffff;
            border: none;
            border-radius: 25px;
            padding: 12px 24px;
            font-size: 1.2em;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #9c27b0;
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(156, 39, 176, 0.5);
        }
        /* Hide Streamlit elements */
        #MainMenu, footer, header, .stDeployButton {
            display: none !important;
        }
                
        /* Hide audio controls */
        audio {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

  # Initialize session state for recording and language selection
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "language" not in st.session_state:
        st.session_state.language = None

    if st.button("Start Recording"):
        st.session_state.recording = True

    if st.session_state.recording:
        st.markdown("""<p color='black'>Recording...</p>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Spanish"):
                st.session_state.language = "Spanish"
                st.write("You selected Spanish.")

        with col2:
            if st.button("French"):
                st.session_state.language = "French"
                st.write("You selected French.")

        if st.session_state.language:
            wav_audio_data = st_audiorec()

            if wav_audio_data is not None:
                # Save the audio data to a temporary file

                with st.spinner("Processing audio..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                        tmpfile.write(wav_audio_data)
                        tmpfile.flush()
                        if st.session_state.language == "Spanish":
                            text, accuracy, fluency = getRating(tmpfile.name, "es-ES")
                        elif st.session_state.language == "French":
                            text, accuracy, fluency = getRating(tmpfile.name, "fr-FR")

                        print(text, accuracy, fluency)
                        gpt(st.session_state.language, text, accuracy, fluency)

                
                st.markdown(f"""
                    <audio controls autoplay>
                        <source src="data:audio/mp3;base64,{base64.b64encode(open("output.mp3", "rb").read()).decode()}" type="audio/mp3">
                    </audio>
                """, unsafe_allow_html=True)




    # File uploader
    # audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])
        # Add your audio processing and prediction code here