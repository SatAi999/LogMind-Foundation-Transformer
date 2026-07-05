import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

class ScaledDotProductAttention(nn.Module):
    """
    Manual Scaled Dot-Product Attention.
    Computes: softmax(Q @ K^T / sqrt(d_k) + Mask) @ V
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Queries of shape (batch_size, num_heads, seq_len, d_k)
            k: Keys of shape (batch_size, num_heads, seq_len, d_k)
            v: Values of shape (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask tensor of shape (batch_size, 1 or num_heads, seq_len, seq_len)
                  containing -1e9 for masked positions and 0 for active positions.
        Returns:
            context: Context representations (batch_size, num_heads, seq_len, d_k)
            attn_weights: Attention weights matrix (batch_size, num_heads, seq_len, seq_len)
        """
        d_k = q.size(-1)
        
        # Calculate attention scores: Q @ K^T
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            # We add the mask (masked positions have very large negative values)
            scores = scores + mask
            
        # Softmax over the last dimension to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attn_weights_drop = self.dropout(attn_weights)
        
        # Compute final context: AttnWeights @ V
        context = torch.matmul(attn_weights_drop, v)
        
        return context, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Manual Multi-Head Self-Attention using low-level tensor projections.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Manual projection weights and biases instead of nn.Linear
        self.w_q = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / math.sqrt(d_model)))
        self.b_q = nn.Parameter(torch.zeros(d_model))
        
        self.w_k = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / math.sqrt(d_model)))
        self.b_k = nn.Parameter(torch.zeros(d_model))
        
        self.w_v = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / math.sqrt(d_model)))
        self.b_v = nn.Parameter(torch.zeros(d_model))
        
        self.w_o = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / math.sqrt(d_model)))
        self.b_o = nn.Parameter(torch.zeros(d_model))
        
        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len, seq_len)
        Returns:
            output: Projected context vector (batch_size, seq_len, d_model)
            attn_weights: Attention maps (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Project inputs to Q, K, V manually
        # x is (B, L, D_model), w_q is (D_model, D_model)
        # Result of matmul(x, w) + b is (B, L, D_model)
        q = torch.matmul(x, self.w_q) + self.b_q
        k = torch.matmul(x, self.w_k) + self.b_k
        v = torch.matmul(x, self.w_v) + self.b_v
        
        # 2. Split into heads: (B, L, H, D_k) and transpose to (B, H, L, D_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply scaled dot-product attention
        context, attn_weights = self.attention(q, k, v, mask)
        
        # 4. Concatenate heads: transpose back to (B, L, H, D_k) and reshape to (B, L, D_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 5. Project back to output space manually
        output = torch.matmul(context, self.w_o) + self.b_o
        
        return output, attn_weights
