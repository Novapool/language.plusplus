import torch

def predict(model, sample_features, word_to_class, device):
    """Predict the word and similarity score for a new sample."""
    model.eval()
    
    with torch.no_grad():
        # Forward pass to get outputs
        word_output, similarity_output = model(sample_features)
        
        # Get predicted word (class with highest probability)
        predicted_class = torch.argmax(word_output, dim=1).item()
        
        # Get similarity score
        predicted_similarity = similarity_output.item()
    
    # Convert class index back to word
    class_to_word = {v: k for k, v in word_to_class.items()}
    predicted_word = class_to_word[predicted_class]
    
    return predicted_word, predicted_similarity