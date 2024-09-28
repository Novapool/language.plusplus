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
        for i, (features, word_labels) in enumerate(train_loader):
            # Move data to GPU if available
            features = features.to(device)
            word_labels = word_labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            word_output, similarity_output = model(features)
            
            # Compute losses
            loss_word = criterion_word(word_output, word_labels)
            # For now, we'll use a dummy target for similarity (all ones)
            similarity_target = torch.ones_like(similarity_output)
            loss_similarity = criterion_similarity(similarity_output, similarity_target)
            
            # Total loss (you can adjust the weights if needed)
            loss = loss_word + 0.1 * loss_similarity
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], '
                      f'Loss: {loss.item():.4f}, Word Loss: {loss_word.item():.4f}, '
                      f'Similarity Loss: {loss_similarity.item():.4f}')
                print(f'GPU Memory Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB')

    print("Training completed.")