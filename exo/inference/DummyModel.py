import numpy as np

class DummyModel:
    def __init__(self, dim, n_heads):
        """
        Initializes the DummyModel with given dimensions and number of heads.
        
        :param dim: The dimensionality of the model's output.
        :param n_heads: The number of attention heads in the model.
        """
        self.dim = dim
        self.n_heads = n_heads

    def __call__(self, input_tensor, start_pos, temperature):
        """
        Simulates a forward pass through the dummy model.
        
        :param input_tensor: The input tensor (e.g., tokenized input).
        :param start_pos: The starting position for inference.
        :param temperature: Temperature parameter for sampling (not used here).
        
        :return: Randomly generated logits shaped like (batch_size, sequence_length, vocab_size).
        """
        batch_size = input_tensor.shape[0]
        sequence_length = input_tensor.shape[1] if len(input_tensor.shape) > 1 else 1
        vocab_size = 128256  # Example vocabulary size
        
        # Generate random logits
        random_logits = np.random.rand(batch_size, sequence_length, vocab_size).astype(np.float32)
        
        # Optionally apply softmax to simulate probabilities
        probabilities = np.exp(random_logits) / np.sum(np.exp(random_logits), axis=-1, keepdims=True)
        
        return probabilities

# Example usage
if __name__ == "__main__":
    # Create a dummy model instance
    dummy_model = DummyModel(dim=4096, n_heads=32)
    
    # Simulate input tensor (e.g., batch size of 2 and sequence length of 5)
    input_tensor = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    
    # Call the dummy model
    output = dummy_model(input_tensor, start_pos=0, temperature=1.0)
    
    print("Output shape:", output.shape)  # Should be (2, 5, 128256)
    print("Output sample:", output[0])     # Print sample output from the first batch
