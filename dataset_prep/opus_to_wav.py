import os
from pydub import AudioSegment

def convert_opus_to_wav(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".opus"):
                opus_path = os.path.join(root, file)
                relative_path = os.path.relpath(opus_path, input_folder)
                wav_path = os.path.join(output_folder, os.path.splitext(relative_path)[0] + ".wav")
                
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                
                audio = AudioSegment.from_file(opus_path, format="opus")
                audio.export(wav_path, format="wav")
                print(f"Converted {opus_path} to {wav_path}")

if __name__ == "__main__":
    input_folder = "dataset"
    output_folder = "dataset_wav"
    convert_opus_to_wav(input_folder, output_folder)