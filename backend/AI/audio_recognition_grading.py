import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from model import MultiOutputRNN
import pickle
import os

def trim_audio(y, sr, threshold_db=-20, min_silence_duration=0.1):
    """Trim silence from the beginning and end of an audio signal."""
    trimmed, _ = librosa.effects.trim(y, top_db=-threshold_db, frame_length=int(sr*min_silence_duration), hop_length=int(sr*min_silence_duration/4))
    return trimmed

def extract_mfcc(y, sr, n_mfcc=13, max_len=100):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def normalize_features(features):
    scaler = StandardScaler()
    features_reshaped = features.reshape(1, -1)
    features_normalized = scaler.fit_transform(features_reshaped)
    return features_normalized.reshape(features.shape)

def process_audio(audio_path, target_duration=3.0):
    # Load and trim the audio
    y, sr = librosa.load(audio_path, sr=None)
    y_trimmed = trim_audio(y, sr)
    
    # Adjust length to target duration
    target_length = int(sr * target_duration)
    if len(y_trimmed) > target_length:
        y_trimmed = y_trimmed[:target_length]
    else:
        y_trimmed = np.pad(y_trimmed, (0, max(0, target_length - len(y_trimmed))))
    
    # Extract MFCC features
    mfcc_features = extract_mfcc(y_trimmed, sr)
    
    # Normalize features
    normalized_features = normalize_features(mfcc_features)
    
    return normalized_features


def load_model(model_path, input_size, hidden_size, num_classes, device):
    model = MultiOutputRNN(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, features, word_to_class, device):
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        word_output, similarity_output = model(features_tensor)
        predicted_class = torch.argmax(word_output, dim=1).item()
        predicted_similarity = similarity_output.item()
    
    class_to_word = {v: k for k, v in word_to_class.items()}
    predicted_word = class_to_word[predicted_class]
    return predicted_word, predicted_similarity

def main(audio_path, model_path, word_to_class_path):
    # Set up device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Process audio
    features = process_audio(audio_path)
    
    # Save processed features
    df = pd.DataFrame({"features": [features], "word": ["unknown"]})
    df.to_pickle("grade_output.pkl")
    print("Processed features saved to grade_output.pkl")

    # Load word_to_class mapping
    with open(word_to_class_path, 'rb') as f:
        word_to_class = pickle.load(f)

    # Load model
    input_size = features.shape[0]
    hidden_size = 128  # Ensure this matches your trained model
    num_classes = len(word_to_class)
    model = load_model(model_path, input_size, hidden_size, num_classes, device)

    # Predict
    predicted_word, similarity_score = predict(model, features, word_to_class, device)

    print(f"Predicted word: {predicted_word}")
    print(f"Similarity score: {similarity_score:.4f}")

if __name__ == "__main__":
    save_dir = os.path.join('backend', 'AI')

    audio_path = "path/to/your/audio.wav"  # Replace with your audio file path
    model_path = os.path.join(save_dir, "trained_model.pth")
    word_to_class_path = os.path.join(save_dir, "word_to_class.pkl")
    
    main(audio_path, model_path, word_to_class_path)