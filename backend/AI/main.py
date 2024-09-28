import torch
from torch.utils.data import DataLoader, TensorDataset
from model import MultiOutputRNN
from training import train_model
from preprocessing import extract_mfcc, normalize_features
from utils import predict

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
input_size = 13  # Number of MFCC coefficients (depends on your feature extraction)
hidden_size = 128  # Number of hidden units in LSTM
num_classes = 10  # Assume we're classifying between 10 words
num_epochs = 10

# Initialize the model and move it to GPU if available
model = MultiOutputRNN(input_size, hidden_size, num_classes).to(device)

# Example: Load audio files, preprocess, and create training data
# This is just a mock example. You need to load your dataset properly.

# Assuming you have a list of audio file paths
audio_paths = ["audio1.wav", "audio2.wav", "audio3.wav"]
X_train = []
word_labels = []  # Example word class labels
similarity_labels = []  # Example similarity scores

for path in audio_paths:
    mfcc_features = extract_mfcc(path)
    mfcc_features = normalize_features(mfcc_features)
    X_train.append(mfcc_features)
    # Append corresponding labels
    word_labels.append(0)  # Just a placeholder
    similarity_labels.append(1.0)  # Just a placeholder

# Convert data to tensors and create a DataLoader
X_train = torch.Tensor(X_train)
word_labels = torch.Tensor(word_labels).long()  # Convert to long for CrossEntropyLoss
similarity_labels = torch.Tensor(similarity_labels)

train_dataset = TensorDataset(X_train, word_labels, similarity_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Train the model
train_model(model, train_loader, num_epochs, device)

# Example: Predict for a new audio sample
test_audio_path = "test_word.wav"
test_mfcc = extract_mfcc(test_audio_path)
test_mfcc = normalize_features(test_mfcc)
test_features = torch.Tensor(test_mfcc).to(device)

predicted_word, predicted_similarity = predict(model, test_features, device)
print(f"Predicted Word: {predicted_word}, Similarity Score: {predicted_similarity}")
