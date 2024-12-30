import torch
import torch.nn as nn
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.transformer_layer import TransformerLayer

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dff, dropout=0.1):
        """
        Implements a transformer encoder with stacked transformer layers.
        Args:
            num_layers (int): Number of transformer layers to stack.
            embed_dim (int): Dimension of input embeddings.
            num_heads (int): Number of attention heads.
            dff (int): Dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, dff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass through the transformer encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
