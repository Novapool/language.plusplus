import librosa
import numpy as np

def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def extract_mel_spectrogram(audio_path, n_mels=128):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)
    return mel_spectrogram_db


audio_path = "data/jugo/common_voice_es_19120092.opus"
mfcc_features = extract_mfcc(audio_path)
mel_spectrogram_features = extract_mel_spectrogram(audio_path)

# print(mfcc_features.shape)
print(mel_spectrogram_features[[128/2, 94/2]])