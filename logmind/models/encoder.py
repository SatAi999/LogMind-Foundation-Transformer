import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from logmind.models.embeddings import LogMindEmbedding
from logmind.models.attention import MultiHeadSelfAttention
from logmind.models.blocks import FeedForwardNetwork, ResidualConnection, LayerNorm

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder block using manual sublayers and residual connections.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, norm_type: str = "pre"):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, activation="gelu", dropout=dropout)
        
        self.res_attn = ResidualConnection(d_model, dropout, norm_type)
        self.res_ffn = ResidualConnection(d_model, dropout, norm_type)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention weights container to extract out of the residual callable closure
        attn_weights = None
        
        def attn_sublayer(h: torch.Tensor) -> torch.Tensor:
            nonlocal attn_weights
            out, weights = self.mha(h, mask)
            attn_weights = weights
            return out
            
        # 1. Multi-Head Attention with residual
        x = self.res_attn(x, attn_sublayer)
        
        # 2. Position-wise Feed-Forward Network with residual
        x = self.res_ffn(x, lambda h: self.ffn(h))
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Full Transformer Encoder Stack composed of embedding layer followed by N manual encoder blocks.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_seq_len: int = 128,
        embed_type: str = "learned",
        norm_type: str = "pre",
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = LogMindEmbedding(vocab_size, d_model, max_seq_len, embed_type, dropout)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout, norm_type)
            for _ in range(n_layers)
        ])
        
        self.norm_type = norm_type
        # Pre-LN Transformer requires a final LayerNorm after all blocks
        if norm_type == "pre":
            self.final_norm = LayerNorm(d_model)
        else:
            self.final_norm = nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
        Returns:
            out: Hidden states of shape (batch_size, seq_len, d_model)
            all_attn_weights: List of attention matrices per layer
        """
        # Compute embeddings: (batch_size, seq_len, d_model)
        h = self.embedding(x)
        
        all_attn_weights = []
        
        # Forward through stack of N blocks
        for block in self.blocks:
            h, attn_weights = block(h, mask)
            all_attn_weights.append(attn_weights)
            
        # Apply final norm if Pre-LN
        out = self.final_norm(h)
        
        return out, all_attn_weights
