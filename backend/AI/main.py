import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import MultiOutputRNN
from training import train_model
from utils import predict

# Path to the preprocessed data
DATA_PATH = './data/'

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def create_word_to_class_mapping(data_path):
    """
    Create a word to class dictionary based on the folder names in the data directory.
    Each folder name represents a word, and it is mapped to a unique class index.
    """
    word_to_class = {}
    for idx, word in enumerate(os.listdir(data_path)):
        word_to_class[word] = idx  # Assign a unique index to each word
    return word_to_class

def load_preprocessed_data(data_path, word_to_class):
    """Load preprocessed data from the directories and assign labels."""
    X_train = []
    word_labels = []
    similarity_labels = []  # Placeholder if needed for similarity scores
    
    # Loop through each word folder
    for word in word_to_class.keys():
        word_folder = os.path.join(data_path, word)
        
        # Loop through each preprocessed sample in the word folder
        for sample_file in os.listdir(word_folder):
            sample_path = os.path.join(word_folder, sample_file)
            
            # Load the preprocessed MFCC data
            mfcc_features = np.load(sample_path)
            X_train.append(mfcc_features)
            
            # Append the corresponding word label (use the class index)
            word_labels.append(word_to_class[word])  # Using class index
            similarity_labels.append(1.0)  # Placeholder, adjust later if needed
    
    # Convert to PyTorch tensors
    X_train = torch.Tensor(X_train)
    word_labels = torch.Tensor(word_labels).long()  # Convert to long for CrossEntropyLoss
    similarity_labels = torch.Tensor(similarity_labels)
    
    return X_train, word_labels, similarity_labels

def main(data_path):
    # Create the word-to-class mapping based on folder names
    word_to_class = create_word_to_class_mapping(data_path)
    print(f"Word to Class Mapping: {word_to_class}")
    
    # Load preprocessed data
    X_train, word_labels, similarity_labels = load_preprocessed_data(data_path, word_to_class)
    
    # Create a DataLoader for training
    train_dataset = TensorDataset(X_train, word_labels, similarity_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Define hyperparameters
    input_size = X_train.shape[2]  # Number of MFCC coefficients
    hidden_size = 128
    num_classes = len(word_to_class)  # Number of classes (based on folders)
    num_epochs = 10
    
    # Initialize the model and move it to GPU if available
    model = MultiOutputRNN(input_size, hidden_size, num_classes).to(device)
    
    # Train the model
    train_model(model, train_loader, num_epochs, device)
    
    # Example: Predict for a new preprocessed sample
    test_sample_path = f'./data/{list(word_to_class.keys())[0]}/sample_1.npy'  # Example for first word in the dictionary
    test_mfcc = np.load(test_sample_path)
    test_features = torch.Tensor(test_mfcc).to(device)
    
    predicted_word, predicted_similarity = predict(model, test_features, device)
    print(f"Predicted Word: {predicted_word}, Similarity Score: {predicted_similarity}")

# Run the main function with the data folder
if __name__ == "__main__":
    main(DATA_PATH)
