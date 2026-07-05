import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

class PairwiseContrastiveLoss(nn.Module):
    """
    Pairwise Siamese Contrastive Loss for Log Sequence Embeddings.
    Pulls sequences of the same category (normal-normal or same root cause failure type)
    together, while pushing different categories apart by a margin.
    Uses vector operations on L2-normalized embeddings (dot product = cosine similarity).
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings of shape (batch_size, d_emb)
            labels: Integer labels of shape (batch_size,) (e.g. RCA label or anomaly label)
        Returns:
            loss: Pairwise contrastive loss scalar
        """
        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Compute cosine similarity matrix: shape (B, B)
        # Since embeddings are L2-normalized, the dot product is exactly the cosine similarity
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Create identity mask to exclude self-similarity (diagonal)
        mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        
        # Pairwise label comparison matrix: True if label[i] == label[j]
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Separate positive pairs (same label) and negative pairs (different labels)
        pos_pairs = label_eq & ~mask
        neg_pairs = ~label_eq & ~mask
        
        loss = torch.tensor(0.0, device=embeddings.device)
        div = 0
        
        if pos_pairs.any():
            # Pull positive pairs together: minimize (1 - similarity)
            loss_pos = (1.0 - sim_matrix)[pos_pairs].mean()
            loss = loss + loss_pos
            div += 1
            
        if neg_pairs.any():
            # Push negative pairs apart: minimize max(0, similarity - margin)
            loss_neg = torch.clamp(sim_matrix - self.margin, min=0.0)[neg_pairs].mean()
            loss = loss + loss_neg
            div += 1
            
        return loss / max(1, div)


class LogMindLoss(nn.Module):
    """
    Unified Loss wrapper for LogMind multi-task learning.
    Configurable to compute individual losses based on the active task mode.
    """
    def __init__(self, ignore_index: int = -100, contrastive_margin: float = 0.5):
        super().__init__()
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.anomaly_loss_fn = nn.BCEWithLogitsLoss()
        self.rca_loss_fn = nn.CrossEntropyLoss()
        self.contrastive_loss_fn = PairwiseContrastiveLoss(margin=contrastive_margin)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        mode: str,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes joint or task-specific losses.
        Args:
            outputs: Outputs dictionary from LogMindModel
            batch: Data batch dictionary containing labels
            mode: "pretrain_mlm", "pretrain_clm", or "finetune"
            weights: Dictionary of loss scaling weights (default is 1.0 for each active head)
        Returns:
            total_loss: Aggregated backpropagation loss scalar
            individual_losses: Dict of float values for monitoring
        """
        if weights is None:
            weights = {"mlm": 1.0, "clm": 1.0, "anomaly": 1.0, "rca": 1.0, "contrastive": 1.0}
            
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=batch["input_ids"].device)
        
        if mode == "pretrain_mlm":
            # MLM loss
            logits = outputs["mlm_logits"] # (B, L, V)
            targets = batch["mlm_labels"]   # (B, L)
            # Reshape to (B*L, V) and (B*L)
            mlm_loss = self.mlm_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss = total_loss + weights["mlm"] * mlm_loss
            loss_dict["mlm_loss"] = mlm_loss.item()
            
        elif mode == "pretrain_clm":
            # Next log event prediction loss
            logits = outputs["mlm_logits"] # (B, L, V)
            targets = batch["clm_labels"]   # (B, L)
            clm_loss = self.mlm_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss = total_loss + weights["clm"] * clm_loss
            loss_dict["clm_loss"] = clm_loss.item()
            
        elif mode == "finetune":
            # 1. Failure/Anomaly Prediction Loss
            anomaly_logits = outputs["anomaly_logits"] # (B,)
            anomaly_targets = batch["anomaly_label"]   # (B,)
            anomaly_loss = self.anomaly_loss_fn(anomaly_logits, anomaly_targets)
            total_loss = total_loss + weights["anomaly"] * anomaly_loss
            loss_dict["anomaly_loss"] = anomaly_loss.item()
            
            # 2. Root Cause Analysis Classification Loss
            rca_logits = outputs["rca_logits"] # (B, num_classes)
            rca_targets = batch["rca_label"]   # (B,)
            rca_loss = self.rca_loss_fn(rca_logits, rca_targets)
            total_loss = total_loss + weights["rca"] * rca_loss
            loss_dict["rca_loss"] = rca_loss.item()
            
            # 3. Contrastive Sequence Embedding Loss
            # We use rca_label as target grouping so that normal sequences align
            # and specific failure types cluster together in vector space.
            embeddings = outputs["embeddings"] # (B, d_emb)
            contrastive_loss = self.contrastive_loss_fn(embeddings, rca_targets)
            total_loss = total_loss + weights["contrastive"] * contrastive_loss
            loss_dict["contrastive_loss"] = contrastive_loss.item()
            
        loss_dict["total_loss"] = total_loss.item()
        
        return total_loss, loss_dict
