import json
import os
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from openai import OpenAI

CONTENT = """So you are a language training assistant. A user will will your 
capabilities to enhance a language that they already know. The user is presented with a couple of 
Spanish words and they pronounce it. That spanish word and their pronunciation is then analyzed by a system.
The system then gives a rating out of 10 on how well the user pronounced the word. After that, you are given that
word and the rating. Assume a threshold of 7. If the user's rating is below 7,you are to correct it.
But do not pronounce the word yet. Just tell them how it is to be pronounced. At the VERY end, tell the user that they will hear a perfect Spanish pronunciation of the word up next.
"""

load_dotenv()

client = OpenAI()

app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    print("Hello, World!")
    return "Hello, World!"


@app.route('/gpt', methods=['POST'])
def gpt():
    data = request.json
    
    score = data["score"]
    word = data["word"]


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

    return jsonify({"message": "Audio files generated successfully"}), 200

@app.route('/audio', methods=['GET'])
def get_audio():
    lang = request.json.get('lang')
    if os.path.exists(f"output_{lang}.mp3"):
        return send_file(f"output_{lang}.mp3", as_attachment=True)
    else:
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == '__main__':
    app.run(port=5000)