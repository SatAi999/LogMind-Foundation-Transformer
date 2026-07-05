import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class MLMHead(nn.Module):
    """
    Masked Language Modeling and Next Log Event Head.
    Projects hidden states of shape (batch_size, seq_len, d_model) to (batch_size, seq_len, vocab_size).
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # Manual projection parameters
        self.w = nn.Parameter(torch.randn(d_model, vocab_size) * (1.0 / math.sqrt(d_model)))
        self.b = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h shape: (batch_size, seq_len, d_model)
        # Output shape: (batch_size, seq_len, vocab_size)
        return torch.matmul(h, self.w) + self.b


class AnomalyHead(nn.Module):
    """
    Sequence classification head for Failure Prediction / Anomaly Detection.
    Maps CLS pooling representation to a single anomaly logit.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # MLP structure: d_model -> d_model -> 1
        self.w1 = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / math.sqrt(d_model)))
        self.b1 = nn.Parameter(torch.zeros(d_model))
        
        self.w2 = nn.Parameter(torch.randn(d_model, 1) * (1.0 / math.sqrt(d_model)))
        self.b2 = nn.Parameter(torch.zeros(1))
        
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        # cls_emb shape: (batch_size, d_model)
        h = self.activation(torch.matmul(cls_emb, self.w1) + self.b1)
        h = self.dropout(h)
        logits = torch.matmul(h, self.w2) + self.b2 # Shape: (batch_size, 1)
        return logits.squeeze(-1) # Shape: (batch_size,)


class RCAHead(nn.Module):
    """
    Sequence-level Root Cause Analysis (RCA) classification head.
    Predicts the multi-class failure category (e.g. 0-5) from CLS pooling representation.
    Designed with a modular MLP so that token-level localization can overlay it later.
    """
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        # MLP: d_model -> d_model -> num_classes
        self.w1 = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / math.sqrt(d_model)))
        self.b1 = nn.Parameter(torch.zeros(d_model))
        
        self.w2 = nn.Parameter(torch.randn(d_model, num_classes) * (1.0 / math.sqrt(d_model)))
        self.b2 = nn.Parameter(torch.zeros(num_classes))
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        # cls_emb shape: (batch_size, d_model)
        h = self.activation(torch.matmul(cls_emb, self.w1) + self.b1)
        h = self.dropout(h)
        logits = torch.matmul(h, self.w2) + self.b2 # Shape: (batch_size, num_classes)
        return logits


class ContrastiveHead(nn.Module):
    """
    Projection head for Contrastive Learning and Incident Similarity Search.
    Projects CLS pooling representation to a normalized lower-dimensional embedding space.
    """
    def __init__(self, d_model: int, d_emb: int = 64):
        super().__init__()
        # Linear projection: d_model -> d_emb
        self.w = nn.Parameter(torch.randn(d_model, d_emb) * (1.0 / math.sqrt(d_model)))
        self.b = nn.Parameter(torch.zeros(d_emb))

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        # cls_emb shape: (batch_size, d_model)
        proj = torch.matmul(cls_emb, self.w) + self.b # Shape: (batch_size, d_emb)
        # L2 Normalize the embeddings
        return F.normalize(proj, p=2, dim=-1)
