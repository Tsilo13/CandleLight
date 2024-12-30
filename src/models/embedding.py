import torch
import torch.nn as nn
import math

class CandlestickEmbedding(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64):
        """
        Converts tokenized candlestick data into embeddings.
        Args:
            input_dim (int): Dimension of input tokens (default 3: [Relative Body, Upper Wick, Lower Wick]).
            embed_dim (int): Dimension of the embedding space.
        """
        super().__init__()
        self.embedding_layer = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass for candlestick embeddings.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        Returns:
            torch.Tensor: Embedded representation of shape (batch_size, sequence_length, embed_dim).
        """
        return self.embedding_layer(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        """
        Adds positional information to embeddings using sinusoidal encodings.
        Args:
            embed_dim (int): Dimension of the embedding space.
            max_len (int): Maximum length of input sequences.
        """
        super().__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        """
        Forward pass for positional encoding.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim).
        Returns:
            torch.Tensor: Positional-encoded tensor of the same shape as input.
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class EmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64):
        """
        Combines candlestick embeddings and positional encodings.
        Args:
            input_dim (int): Dimension of input tokens.
            embed_dim (int): Dimension of the embedding space.
        """
        super().__init__()
        self.embedding = CandlestickEmbedding(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)

    def forward(self, x):
        """
        Forward pass for combined embeddings and positional encodings.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        Returns:
            torch.Tensor: Combined embedding and positional encoding.
        """
        x = self.embedding(x)
        return self.positional_encoding(x)
