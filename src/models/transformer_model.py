import torch
import torch.nn as nn
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.embedding import EmbeddingWithPositionalEncoding
from src.models.transformer_encoder import TransformerEncoder

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, dff, num_layers, output_dim, dropout=0.1):
        """
        Full transformer model for predicting candlestick patterns.
        Args:
            input_dim (int): Dimension of input tokens (e.g., 3 for candlesticks).
            embed_dim (int): Dimension of embedding space.
            num_heads (int): Number of attention heads.
            dff (int): Dimension of feed-forward network.
            num_layers (int): Number of transformer layers.
            output_dim (int): Dimension of the model output (e.g., 3 for relative body, upper wick, lower wick).
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embedding = EmbeddingWithPositionalEncoding(input_dim, embed_dim)
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, dff, dropout)
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the full transformer model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        # Embed the input
        x = self.embedding(x)
        
        # Transformer expects (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        
        # Pass through the encoder
        x = self.encoder(x)
        
        # Back to (batch_size, seq_len, embed_dim)
        x = x.permute(1, 0, 2)
        
        # Generate output predictions
        return self.output_layer(x)
