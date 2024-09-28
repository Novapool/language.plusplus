import librosa
import numpy as np
import os
import pandas as pd

def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
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
    df.to_pickle("features.pkl")


audio_path = "data/"

extract_features(audio_path)
# mfcc_features = extract_mfcc(audio_path)
# mel_spectrogram_features = extract_mel_spectrogram(audio_path)

# print(mfcc_features.shape)