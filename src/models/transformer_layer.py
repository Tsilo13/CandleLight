import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dff, dropout=0.1):
        """
        Implements a single transformer encoder layer.
        Args:
            embed_dim (int): Dimension of input embeddings.
            num_heads (int): Number of attention heads.
            dff (int): Dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dff),
            nn.ReLU(),
            nn.Linear(dff, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the transformer layer.
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x
