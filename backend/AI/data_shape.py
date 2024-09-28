import pandas as pd
import torch

# Path to the .pkl file
PKL_FILE = 'data.pkl'

# Load the .pkl file and return the features and word labels
def load_data_from_pkl(pkl_file):
    df = pd.read_pickle(pkl_file)

    return df

# Load the data
df = load_data_from_pkl(PKL_FILE)

features = df["features"].tolist()
labels = df["word"].tolist()

# Convert features to PyTorch tensors
X_train = [torch.Tensor(feature) for feature in features]

# Stack the tensors to create a batch
X_train = torch.stack(X_train)

# Print the shape of X_train to check dimensions
print(f"Shape of X_train: {X_train.shape}")
