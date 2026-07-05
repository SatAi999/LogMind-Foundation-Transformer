import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
from logmind.models.encoder import TransformerEncoder
from logmind.models.heads import MLMHead, AnomalyHead, RCAHead, ContrastiveHead

class LogMindModel(nn.Module):
    """
    Combined LogMind Multi-task Foundation Model.
    Integrates a shared Transformer Encoder with modular heads:
    - MLM / Next Event Prediction Head
    - Failure / Anomaly Classification Head
    - Sequence-level Root Cause Analysis (RCA) classification head
    - Incident Similarity Search Contrastive Embedding Head
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 3,
        max_seq_len: int = 128,
        num_classes_anomaly: int = 1,
        num_classes_rca: int = 6,
        d_emb: int = 64,
        embed_type: str = "learned",
        norm_type: str = "pre",
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Shared Transformer Encoder Stack
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            embed_type=embed_type,
            norm_type=norm_type,
            dropout=dropout
        )
        
        # Modular Multi-task heads
        self.mlm_head = MLMHead(d_model, vocab_size)
        self.anomaly_head = AnomalyHead(d_model, dropout)
        self.rca_head = RCAHead(d_model, num_classes_rca, dropout)
        self.contrastive_head = ContrastiveHead(d_model, d_emb)

    def get_attention_mask(self, padding_mask: torch.Tensor, mask_type: str) -> torch.Tensor:
        """
        Dynamically constructs an attention mask matrix at runtime.
        Args:
            padding_mask: Boolean tensor of shape (batch_size, seq_len)
                          where True is active and False is pad.
            mask_type: "bidirectional" (for representation learning) or
                       "causal" (for next-event prediction and generation)
        Returns:
            mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
                  containing 0.0 for active connections and -1e9 for masked connections.
        """
        batch_size, seq_len = padding_mask.size()
        
        # Pad mask matrix: shape (batch_size, 1, 1, seq_len)
        # Broadcasting will ensure we mask out column j if key j is padding.
        pad_mask_matrix = padding_mask.unsqueeze(1).unsqueeze(2)
        
        if mask_type == "bidirectional":
            mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=padding_mask.device)
            mask = mask.masked_fill(~pad_mask_matrix, -1e9)
            return mask
            
        elif mask_type == "causal":
            # Causal mask: lower triangular matrix of shape (seq_len, seq_len)
            causal_tril = torch.tril(torch.ones(seq_len, seq_len, device=padding_mask.device)).bool()
            
            # Combine pad mask matrix (B, 1, 1, L) and causal mask (1, 1, L, L)
            combined = pad_mask_matrix & causal_tril.unsqueeze(0).unsqueeze(1)
            
            mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=padding_mask.device)
            mask = mask.masked_fill(~combined, -1e9)
            return mask
            
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}. Must be 'bidirectional' or 'causal'.")

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        mask_type: str = "bidirectional",
        run_heads: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            padding_mask: Boolean tensor of shape (batch_size, seq_len)
            mask_type: "bidirectional" or "causal" attention masking
            run_heads: Optional list of heads to execute to save computation:
                       ["mlm", "anomaly", "rca", "contrastive"] (None runs all heads)
        Returns:
            outputs: Dictionary of outputs and intermediate representations
        """
        # 1. Construct attention mask
        attn_mask = self.get_attention_mask(padding_mask, mask_type)
        
        # 2. Run shared encoder
        encoder_states, attn_maps = self.encoder(input_ids, attn_mask)
        
        # 3. Pooling: Extract representation of the [CLS] token (index 0)
        cls_emb = encoder_states[:, 0, :]
        
        outputs = {
            "encoder_states": encoder_states,
            "cls_embedding": cls_emb,
            "attention_maps": attn_maps
        }
        
        # 4. Conditionally run multi-task prediction heads
        if run_heads is None or "mlm" in run_heads:
            outputs["mlm_logits"] = self.mlm_head(encoder_states)
            
        if run_heads is None or "anomaly" in run_heads:
            outputs["anomaly_logits"] = self.anomaly_head(cls_emb)
            
        if run_heads is None or "rca" in run_heads:
            outputs["rca_logits"] = self.rca_head(cls_emb)
            
        if run_heads is None or "contrastive" in run_heads:
            outputs["embeddings"] = self.contrastive_head(cls_emb)
            
        return outputs
