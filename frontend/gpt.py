
from dotenv import load_dotenv
from openai import OpenAI

CONTENT = """So you are a language training assistant. A user will will your 
capabilities to enhance a language that they already know. The user is presented with a couple of 
Spanish words and they pronounce it. That spanish word and their pronunciation is then analyzed by a system.
The system then gives a rating out of 10 on how well the user pronounced the word. After that, you are given that
word and the rating. Assume a threshold of 7. If the user's rating is below 7,you are to correct it.
But do not pronounce the word yet. Just tell them how it is to be pronounced. You are ONLY to speak in Spanish."""

load_dotenv()

client = OpenAI()

def gpt(word, score):
    completion_eng = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CONTENT},
            {
                "role": "user",
                "content": f"Word: [{word}]. Pronunciation: [{word}]. Rating: {score}."
            }
        ]
    )

    completion_es = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a word pronunciator. Just pronounce the given spanish word in spanish."},
            {
                "role": "user",
                "content": word
            }
        ]
    )

    for lang in ["eng", "es"]:
        completion = completion_eng if lang == "eng" else completion_es
        
        msg = (completion.choices[0].message.content)

        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=msg,
        )

        response.stream_to_file(f"output_{lang}.mp3")

    return "output_eng.mp3", "output_es.mp3"