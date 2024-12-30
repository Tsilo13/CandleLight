import torch
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the embedding module
from src.models.embedding import EmbeddingWithPositionalEncoding

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected. Please check your CUDA installation.")

# Instantiate and move the embedding model to GPU
embedding_model = EmbeddingWithPositionalEncoding(input_dim=3, embed_dim=64).to(device)

# Create dummy data and move to GPU
sample_data = torch.rand(32, 48, 3).to(device)  # Batch size=32, sequence length=48, features=3
print("Sample data device:", sample_data.device)

# Forward pass (GPU test)
embedded_data = embedding_model(sample_data)
print("Shape of embedded data:", embedded_data.shape)
