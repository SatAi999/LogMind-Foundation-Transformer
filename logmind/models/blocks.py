import torch
import torch.nn as nn
import math
from typing import Callable, Optional

class LayerNorm(nn.Module):
    """
    Manual Layer Normalization implemented using low-level PyTorch math.
    Normalizes across the last dimension.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance along the last feature dimension (-1)
        mean = x.mean(dim=-1, keepdim=True)
        # Using biased variance (unbiased=False) matches PyTorch's native LayerNorm
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize: (x - mean) / sqrt(var + eps)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable scale (gamma) and shift (beta)
        return self.gamma * x_norm + self.beta


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network implemented with low-level parameter projections.
    """
    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu", dropout: float = 0.1):
        super().__init__()
        
        # Linear layer 1 parameters: d_model -> d_ff
        self.w1 = nn.Parameter(torch.randn(d_model, d_ff) * (1.0 / math.sqrt(d_model)))
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        
        # Linear layer 2 parameters: d_ff -> d_model
        self.w2 = nn.Parameter(torch.randn(d_ff, d_model) * (1.0 / math.sqrt(d_ff)))
        self.b2 = nn.Parameter(torch.zeros(d_model))
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # First projection: x @ w1 + b1
        h = self.activation(torch.matmul(x, self.w1) + self.b1)
        h = self.dropout(h)
        
        # Second projection: h @ w2 + b2
        out = torch.matmul(h, self.w2) + self.b2
        return out


class ResidualConnection(nn.Module):
    """
    Residual connection wrapper that manages skip connections and normalization.
    Supports both Pre-LayerNorm (GPT/Modern standard) and Post-LayerNorm (Classic BERT).
    """
    def __init__(self, d_model: int, dropout: float = 0.1, norm_type: str = "pre"):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_type = norm_type
        
        assert norm_type in ["pre", "post"], "norm_type must be either 'pre' or 'post'"

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Applies residual connection with sublayer mapping.
        For Pre-LN: x + Sublayer(Norm(x))
        For Post-LN: Norm(x + Sublayer(x))
        """
        if self.norm_type == "pre":
            # Apply normalization before sending to sublayer
            sublayer_out = sublayer(self.norm(x))
            return x + self.dropout(sublayer_out)
        else:
            # Apply normalization after adding residual
            sublayer_out = sublayer(x)
            return self.norm(x + self.dropout(sublayer_out))
