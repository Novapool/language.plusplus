import os
import librosa
from datasets import Dataset

def load_dataset(data_dir):
    data = []
    for word in os.listdir(data_dir):
        word_dir = os.path.join(data_dir, word)
        if os.path.isdir(word_dir):
            for file in os.listdir(word_dir):
                if file.endswith(".wav"):
                    file_path = os.path.join(word_dir, file)
                    audio, sr = librosa.load(file_path, sr=16000)
                    data.append({
                        "audio": {"array": audio, "sampling_rate": sr},
                        "text": word,
                    })
    return Dataset.from_dict({k: [d[k] for d in data] for k in data[0]})

if __name__ == "__main__":
    dataset = load_dataset("dataset_wav")
    dataset.save_to_disk("prepared_dataset")
    print(f"Dataset prepared and saved with {len(dataset)} samples.")