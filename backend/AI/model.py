import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiOutputRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiOutputRNN, self).__init__()
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layer for word recognition (classification)
        self.fc_word = nn.Linear(hidden_size, num_classes)
        
        # Fully connected layer for similarity score (regression)
        self.fc_similarity = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Pass through LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out_last = lstm_out[:, -1, :]
        
        # Apply dropout for regularization
        lstm_out_last = self.dropout(lstm_out_last)
        
        # Word recognition output (classification)
        word_output = self.fc_word(lstm_out_last)
        word_output = F.softmax(word_output, dim=1)
        
        # Similarity score output (regression)
        similarity_output = self.fc_similarity(lstm_out_last)
        similarity_output = torch.sigmoid(similarity_output)
        
        return word_output, similarity_output