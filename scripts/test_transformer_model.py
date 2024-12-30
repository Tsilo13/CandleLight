import torch
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.transformer_model import TransformerModel

# Parameters
input_dim = 3        # Candlestick features: [Relative Body, Upper Wick, Lower Wick]
embed_dim = 64       # Embedding dimension
num_heads = 4        # Number of attention heads
dff = 256            # Feed-forward network dimension
num_layers = 4       # Number of transformer layers
output_dim = 3       # Predicting [Relative Body, Upper Wick, Lower Wick]
seq_len = 48         # Sequence length
batch_size = 32      # Batch size

# Initialize the model
model = TransformerModel(input_dim, embed_dim, num_heads, dff, num_layers, output_dim)

# Dummy input data
dummy_data = torch.rand(batch_size, seq_len, input_dim)

# Forward pass
output = model(dummy_data)
print("Shape of model output:", output.shape)  # Expected: (batch_size, seq_len, output_dim)
