import streamlit as st
import numpy as np
import tempfile
import os
import base64

from gpt import gpt

from st_audiorec import st_audiorec

from audio_recorder_streamlit import audio_recorder

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
            background-color: #f0f2f6;
            color: #000000;
            margin: 0;
            padding: 0;
        }
        .stApp {
            background-color: #f0f2f6;
        }
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: calc(100vh - 60px);
            padding-top: 60px;
        }
        .content {
            text-align: center;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .title {
            font-size: 3em;
            font-weight: 700;
            color: #6a1b9a;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .subtitle {
            font-size: 1.5em;
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
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }
        .stButton > button {
            background-color: #6a1b9a;
            color: #ffffff;
            border: none;
            border-radius: 25px;
            padding: 10px 20px;
            font-size: 1em;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 120px;
        }
        .stButton > button:hover {
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
        /* Top bar styles */
        .top-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background-color: #6a1b9a;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .top-bar-title {
            color: #ffffff;
            font-size: 1.5em;
            font-weight: 700;
        }
        </style>
    """, unsafe_allow_html=True)

    # Add the top bar
    st.markdown("""
        <div class="top-bar">
            <div class="top-bar-title">Language++</div>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state for recording and language selection
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "language" not in st.session_state:
        st.session_state.language = None

    st.markdown("<h1 class='title'>Language Practice App</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Improve your pronunciation skills</p>", unsafe_allow_html=True)

    if not st.session_state.recording:
        if st.button("Start Recording", key="start_recording"):
            st.session_state.recording = True

    if st.session_state.recording:
        st.markdown("<h2 class='subtitle'>Select a language:</h2>", unsafe_allow_html=True)
        
        # Define language options
        languages = {
            "Spanish": "es-ES",
            "French": "fr-FR",
            "Korean": "ko-KR",
            "Japanese": "ja-JP",
            "Russian": "ru-RU",
            "Arabic": "ar-SA",
            "German": "de-DE",
            "Italian": "it-IT"
        }

        # Create a 4x2 grid for language buttons
        cols = st.columns(4)
        for i, (lang, code) in enumerate(languages.items()):
            with cols[i % 4]:
                if st.button(lang, key=f"{lang.lower()}_button"):
                    st.session_state.language = lang
                    st.session_state.language_code = code

        if st.session_state.language:
            st.write(f"You selected {st.session_state.language}.")
            st.markdown("<p class='description'>Please start speaking to record your voice.</p>", unsafe_allow_html=True)
            
            # wav_audio_data = st_audiorec()

            audio_bytes = audio_recorder(text="", icon_size="3x", icon_name="microphone-lines")
            if audio_bytes:
            #     st.audio(audio_bytes, format="audio/wav")

            # if wav_audio_data is not None:
                with st.spinner("Processing audio..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                        tmpfile.write(audio_bytes)
                        tmpfile.flush()
                        text, accuracy, fluency = getRating(tmpfile.name, st.session_state.language_code)

                        print(text, accuracy, fluency)
                        gpt(st.session_state.language, text, accuracy, fluency)

                st.markdown(f"""
                    <audio controls autoplay>
                        <source src="data:audio/mp3;base64,{base64.b64encode(open("output.mp3", "rb").read()).decode()}" type="audio/mp3">
                    </audio>
                """, unsafe_allow_html=True)

    # File uploader (commented out as per the original code)
    # audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])