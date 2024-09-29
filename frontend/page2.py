import streamlit as st

def page2():
    st.title("Upload Audio")
    st.write("Upload your audio file to get started with Language++.")

    # File uploader
    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        st.write("File uploaded successfully!")
        # Add your audio processing and prediction code here