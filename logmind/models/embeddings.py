import torch
import torch.nn as nn
import math
from typing import Optional

class TokenEmbedding(nn.Module):
    """
    Manual token embedding layer using low-level PyTorch parameter indexing.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # Initialize weights with standard normal distribution scaled by 0.02 (BERT standard)
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        # Output shape: (batch_size, seq_len, d_model)
        return self.weight[x]


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (non-learnable).
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension shape: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Returns pe sliced to seq_len
        return self.pe[:, :seq_len]


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embedding layer (GPT/BERT style).
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Returns weight sliced to seq_len and broadcasted
        return self.weight[:seq_len].unsqueeze(0)


class LogMindEmbedding(nn.Module):
    """
    LogMind combined embedding layer containing:
    - Token Embedding
    - Positional Embedding (Sinusoidal or Learned)
    - Dropout
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 128, 
                 embed_type: str = "learned", dropout: float = 0.1):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        
        if embed_type == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEncoding(d_model, max_len)
        elif embed_type == "learned":
            self.pos_embed = LearnedPositionalEmbedding(d_model, max_len)
        else:
            raise ValueError(f"Unknown embed_type: {embed_type}")
            
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        x_embed = self.token_embed(x)
        p_embed = self.pos_embed(x_embed)
        
        # Sum token embeddings and positional encodings
        return self.dropout(x_embed + p_embed)
