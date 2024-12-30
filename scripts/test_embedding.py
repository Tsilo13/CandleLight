import torch
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the embedding module
from src.models.embedding import EmbeddingWithPositionalEncoding

# Initialize the embedding module
embedding_model = EmbeddingWithPositionalEncoding(input_dim=3, embed_dim=64)

# Create dummy data
sample_data = torch.rand(32, 48, 3)  # (batch_size=32, sequence_length=48, input_dim=3)

# Get embeddings
embedded_data = embedding_model(sample_data)

print("Shape of embedded data:", embedded_data.shape)  # Should output: (32, 48, 64)
