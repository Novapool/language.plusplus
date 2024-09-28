import librosa
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_mfcc(audio_path, n_mfcc=13, max_len=100):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate to ensure consistent length
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def extract_mel_spectrogram(audio_path, n_mels=128):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)
    return mel_spectrogram_db

def extract_features(audio_dir_path):

    data = []
    words = os.listdir(audio_dir_path)

    for word in words:
        word_path = os.path.join(audio_dir_path, word)
        audio_files = os.listdir(word_path)

        for audio_file in audio_files:
            audio_path = os.path.join(word_path, audio_file)
            mfcc_features = extract_mfcc(audio_path)
            data.append((mfcc_features, word))

    df = pd.DataFrame(data, columns=["features", "word"])

    return df

def normalize_features(df):
    scaler = StandardScaler()
    # Extract features and reshape for normalization
    features = np.array(df["features"].tolist())
    num_samples, num_features, num_frames = features.shape
    features_reshaped = features.reshape(num_samples, -1)
    # Normalize features
    features_normalized = scaler.fit_transform(features_reshaped)
    # Reshape back to original shape
    features_normalized = features_normalized.reshape(num_samples, num_features, num_frames)
    # Create a new DataFrame with normalized features
    df_normalized = pd.DataFrame({"features": list(features_normalized), "word": df["word"]})
    return df_normalized


audio_path = "data/"

df = extract_features(audio_path)
df_normalized = normalize_features(df)
df_normalized.to_pickle("data.pkl")

# mfcc_features = extract_mfcc(audio_path)
# mel_spectrogram_features = extract_mel_spectrogram(audio_path)

# print(mfcc_features.shape)