# Language++

An AI-Assisted Language Trainer

Welcome to your very own language pronunciation trainer. This app is built on top on modern AI technologies and leverages mutliform forms of AI to analyze pronunciation, sentiment analysis, speaking flow, coherence, and gives actually useful feedback in a conversational manner. There is no text input or output. It's like a human trainer! You talk to it as if it's a real person teaching you how to get better on a new language!

## Tech Stack

- Streamlit! The entire app is coded in streamlit along with its AI components and hosted on Streamlit Community Cloud
- Microsoft Azure AI: The app utilizes Azure AI's pronunciation assessment to break down a user's spoken phrase
- Open AI GPT 4-o: It uses the latest GPT-4o for advanced text generation and text-to-speech in a natural voice.

## AI Pipeline

- The user speaks into the app
- Azure AI is used for pronunciation and fluency analysis
- The scores are sent to GPT-4o for textual analysis.
- GPT-4o generates feedback
- OpenAI Text-to-Speech (TTS) is used to convert the textual feedback into natural-sounding audio feedback
- The process continues into a conversation.
