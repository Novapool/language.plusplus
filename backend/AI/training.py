import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, num_epochs, device):
    # Loss functions
    criterion_word = nn.CrossEntropyLoss()
    criterion_similarity = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    
    for epoch in range(num_epochs):
        for i, (features, word_labels, similarity_labels) in enumerate(train_loader):
            # Move data to GPU if available
            features = features.to(device)
            word_labels = word_labels.to(device)
            similarity_labels = similarity_labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            word_output, similarity_output = model(features)
            
            # Compute losses
            loss_word = criterion_word(word_output, word_labels)
            loss_similarity = criterion_similarity(similarity_output.squeeze(), similarity_labels)
            
            # Total loss
            loss = loss_word + loss_similarity
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item():.4f}')
