import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import MultiOutputRNN
from training import train_model
from utils import predict

# Path to the preprocessed data
PKL_FILE = 'data.pkl'

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_data_from_pkl(pkl_file):
    """Load the .pkl file and return the features and words."""
    df = pd.read_pickle(pkl_file)
    return df['features'].tolist(), df['word'].tolist()

def create_word_to_class_mapping(words):
    """
    Create a word to class dictionary based on the unique words in the dataset.
    Each word is mapped to a unique class index.
    """
    word_to_class = {word: idx for idx, word in enumerate(sorted(set(words)))}
    return word_to_class

def prepare_data(features, words, word_to_class):
    """
    Convert features and words into PyTorch tensors with numerical labels.
    Features are kept as is, and words are converted to class indices.
    """
    X_train = [torch.tensor(feature, dtype=torch.float) for feature in features]
    word_labels = [word_to_class[word] for word in words]
    
    X_train = torch.stack(X_train)
    word_labels = torch.tensor(word_labels, dtype=torch.long)
    
    return X_train, word_labels

def main(pkl_file):
    # Load the data from the .pkl file
    features, words = load_data_from_pkl(pkl_file)
    
    # Create the word-to-class mapping
    word_to_class = create_word_to_class_mapping(words)
    print(f"Number of unique words: {len(word_to_class)}")
    
    # Prepare the data for training
    X_train, word_labels = prepare_data(features, words, word_to_class)
    
    # Create a DataLoader for training
    train_dataset = TensorDataset(X_train, word_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Define hyperparameters
    input_size = 100  # Number of MFCC coefficients
    hidden_size = 128
    num_classes = len(word_to_class)  # Number of unique words in the dataset
    num_epochs = 100
    
    # Initialize the model and move it to GPU if available
    model = MultiOutputRNN(input_size, hidden_size, num_classes).to(device)
    
    # Train the model
    train_model(model, train_loader, num_epochs, device)
    
    # Example: Predict for a new sample
    test_features = torch.tensor(features[0], dtype=torch.float).unsqueeze(0).to(device)
    predicted_word, predicted_similarity = predict(model, test_features, word_to_class, device)
    print(f"Predicted Word: {predicted_word}, Similarity Score: {predicted_similarity:.4f}")

# Run the main function with the .pkl file
if __name__ == "__main__":
    main(PKL_FILE)