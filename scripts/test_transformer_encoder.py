import torch
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer_encoder import TransformerEncoder

# Parameters
num_layers = 4
embed_dim = 64
num_heads = 4
dff = 256
seq_len = 48
batch_size = 32

# Initialize transformer encoder
encoder = TransformerEncoder(num_layers, embed_dim, num_heads, dff)

# Dummy data
dummy_input = torch.rand(seq_len, batch_size, embed_dim)

# Forward pass
output = encoder(dummy_input)
print("Shape of output:", output.shape)  # Should be (seq_len, batch_size, embed_dim)
