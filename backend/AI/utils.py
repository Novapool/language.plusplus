import torch

def predict(model, sample_features, device):
    """Predict the word and similarity score for a new sample."""
    model.eval()
    
    with torch.no_grad():
        # Move features to GPU if available
        sample_features = sample_features.to(device)
        
        # Forward pass to get outputs
        word_output, similarity_output = model(sample_features.unsqueeze(0))  # Add batch dimension
        
        # Get predicted word (class with highest probability)
        predicted_word = torch.argmax(word_output, dim=1).item()
        
        # Get similarity score
        predicted_similarity = similarity_output.item()
    
    return predicted_word, predicted_similarity
