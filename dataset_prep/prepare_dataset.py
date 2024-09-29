import os
import librosa
from datasets import Dataset
from sklearn.model_selection import train_test_split

def load_dataset(data_dir):
    data = []
    for word in os.listdir(data_dir):
        word_dir = os.path.join(data_dir, word)
        if os.path.isdir(word_dir):
            word_samples = []
            for file in os.listdir(word_dir):
                if file.endswith(".wav"):
                    file_path = os.path.join(word_dir, file)
                    audio, sr = librosa.load(file_path, sr=16000)
                    word_samples.append({
                        "audio": {"array": audio, "sampling_rate": sr},
                        "text": word,
                    })
            # Limit to 50 samples per word, if more exist
            if len(word_samples) > 50:
                word_samples = word_samples[:50]
            data.extend(word_samples)
    
    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=[d["text"] for d in data])
    
    train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})
    test_dataset = Dataset.from_dict({k: [d[k] for d in test_data] for k in test_data[0]})
    
    return {"train": train_dataset, "test": test_dataset}

if __name__ == "__main__":
    dataset = load_dataset("dataset_wav")
    dataset.save_to_disk("prepared_dataset")
    print(f"Dataset prepared and saved with {len(dataset['train'])} training samples and {len(dataset['test'])} test samples.")