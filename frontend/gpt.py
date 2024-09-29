from openai import OpenAI
import streamlit as st

CONTENT = """So you are a language training assistant. A user will will your 
capabilities to enhance a language that they already know. The user has been presented with a phrase
in a certain language and you will be told what that language is. The user will pronounce the phrases and then
we have our AI infrastructure analyze the pronunciation and give it an accuracy and fluency score. You will be given the 
phrase, the language it is in, the accuracy score and the fluency score. If either of them are below 85%, then 
give them some helpful tips of how to improve based on their specific custom pronunciation.
Then at the end, you are to continue providing them with another phrase in the same language. Make
the difficulty of the pronunciation depend on the user's previous performance."""

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

client = OpenAI()

def gpt(lang, text, accuracy, fluency):
    completion_eng = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CONTENT},
            {
                "role": "user",
                "content": f"Text: [{text}]. Language: [{lang}]. Accuracy: {accuracy}. Fluency: {fluency}."
            }
        ]
    )

    # completion_es = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": "You are a word pronunciator. Just pronounce the given spanish word in spanish."},
    #         {
    #             "role": "user",
    #             "content": word
    #         }
    #     ]
    # )

    completion = completion_eng
    
    msg = (completion.choices[0].message.content)

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=msg,
    )

    response.stream_to_file(f"output.mp3")

    return "output.mp3"