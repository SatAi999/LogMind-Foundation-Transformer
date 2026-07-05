import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from logmind.models.logmind_model import LogMindModel
from logmind.training.losses import LogMindLoss
from logmind.training.metrics import (
    compute_binary_metrics, 
    compute_multiclass_metrics, 
    compute_language_modeling_metrics
)

logger = logging.getLogger(__name__)

class LogMindTrainer:
    """
    Unified training pipeline for LogMind foundation model.
    Supports:
    - Pretraining: MLM (bidirectional) and Next Event CLM (causal)
    - Fine-tuning: Failure Prediction (binary), RCA (multiclass), and Contrastive Embeddings
    """
    def __init__(
        self,
        model: LogMindModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: LogMindLoss,
        device: torch.device,
        config: Dict[str, Any],
        scheduler: Optional[Any] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Loss scale weights for multi-task learning
        self.loss_weights = {
            "mlm": 1.0,
            "clm": 1.0,
            "anomaly": 1.0,
            "rca": 1.0,
            "contrastive": 0.5
        }

    def train_epoch(self, epoch: int, mode: str) -> Dict[str, float]:
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        individual_losses: Dict[str, float] = {}
        
        # Language modeling variables
        correct_tokens = 0
        total_tokens = 0
        
        # Determine attention mask type to use
        mask_type = "causal" if mode == "pretrain_clm" else "bidirectional"
        
        # Determine which heads are required to save compute
        run_heads = ["mlm"] if "pretrain" in mode else ["anomaly", "rca", "contrastive"]
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            inputs = batch["input_ids"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            
            # Move targets to device
            batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs, padding_mask, mask_type=mask_type, run_heads=run_heads)
            
            # Loss computation
            loss, loss_dict = self.loss_fn(outputs, batch_device, mode, self.loss_weights)
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping to stabilize training
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
                
            total_loss += loss.item()
            for k, v in loss_dict.items():
                individual_losses[k] = individual_losses.get(k, 0.0) + v
                
            # Log event accuracy calculations for pretraining monitor
            if mode == "pretrain_mlm" and "mlm_logits" in outputs:
                # Calculate correct masked predictions
                logits = outputs["mlm_logits"]
                targets = batch_device["mlm_labels"]
                mask = targets != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    correct_tokens += (preds[mask] == targets[mask]).sum().item()
                    total_tokens += mask.sum().item()
                    
            elif mode == "pretrain_clm" and "mlm_logits" in outputs:
                # Calculate correct next token predictions
                logits = outputs["mlm_logits"]
                targets = batch_device["clm_labels"]
                mask = targets != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    correct_tokens += (preds[mask] == targets[mask]).sum().item()
                    total_tokens += mask.sum().item()

        # Average losses
        num_batches = len(self.train_loader)
        avg_metrics = {k: v / num_batches for k, v in individual_losses.items()}
        avg_metrics["loss"] = total_loss / num_batches
        
        if total_tokens > 0:
            avg_metrics["token_accuracy"] = correct_tokens / total_tokens
            
        return avg_metrics

    @torch.no_grad()
    def validate(self, mode: str) -> Dict[str, float]:
        """
        Runs validation and returns metrics.
        """
        self.model.eval()
        total_loss = 0.0
        individual_losses: Dict[str, float] = {}
        
        # Ground truths and predictions lists for scoring
        all_anomaly_true = []
        all_anomaly_pred = []
        
        all_rca_true = []
        all_rca_pred = []
        
        correct_tokens = 0
        total_tokens = 0
        
        mask_type = "causal" if mode == "pretrain_clm" else "bidirectional"
        run_heads = ["mlm"] if "pretrain" in mode else ["anomaly", "rca", "contrastive"]
        
        for batch in self.val_loader:
            inputs = batch["input_ids"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = self.model(inputs, padding_mask, mask_type=mask_type, run_heads=run_heads)
            
            loss, loss_dict = self.loss_fn(outputs, batch_device, mode, self.loss_weights)
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                individual_losses[k] = individual_losses.get(k, 0.0) + v
                
            # Collect outputs for metric calculations
            if mode == "pretrain_mlm" and "mlm_logits" in outputs:
                logits = outputs["mlm_logits"]
                targets = batch_device["mlm_labels"]
                mask = targets != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    correct_tokens += (preds[mask] == targets[mask]).sum().item()
                    total_tokens += mask.sum().item()
                    
            elif mode == "pretrain_clm" and "mlm_logits" in outputs:
                logits = outputs["mlm_logits"]
                targets = batch_device["clm_labels"]
                mask = targets != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    correct_tokens += (preds[mask] == targets[mask]).sum().item()
                    total_tokens += mask.sum().item()
                    
            elif mode == "finetune":
                # Anomaly classification predictions (probability via sigmoid)
                anomaly_probs = torch.sigmoid(outputs["anomaly_logits"])
                all_anomaly_true.extend(batch["anomaly_label"].cpu().numpy())
                all_anomaly_pred.extend(anomaly_probs.cpu().numpy())
                
                # RCA classification predictions
                rca_logits = outputs["rca_logits"]
                all_rca_true.extend(batch["rca_label"].cpu().numpy())
                all_rca_pred.extend(rca_logits.cpu().numpy())

        # Average losses
        num_batches = len(self.val_loader)
        val_metrics = {f"val_{k}": v / num_batches for k, v in individual_losses.items()}
        val_metrics["val_loss"] = total_loss / num_batches
        
        # Calculate scores
        if "pretrain" in mode:
            if total_tokens > 0:
                avg_lm_loss = val_metrics.get("val_mlm_loss", val_metrics.get("val_clm_loss", val_metrics["val_loss"]))
                lm_metrics = compute_language_modeling_metrics(avg_lm_loss, correct_tokens, total_tokens)
                val_metrics["val_token_accuracy"] = lm_metrics["accuracy"]
                val_metrics["val_perplexity"] = lm_metrics["perplexity"]
                
        elif mode == "finetune":
            # Binary failure prediction metrics
            anom_metrics = compute_binary_metrics(
                np.array(all_anomaly_true), 
                np.array(all_anomaly_pred)
            )
            for k, v in anom_metrics.items():
                val_metrics[f"val_anomaly_{k}"] = v
                
            # RCA classification metrics
            rca_metrics = compute_multiclass_metrics(
                np.array(all_rca_true), 
                np.array(all_rca_pred)
            )
            for k, v in rca_metrics.items():
                val_metrics[f"val_rca_{k}"] = v
                
        return val_metrics

    def fit(self, epochs: int, mode: str) -> List[Dict[str, float]]:
        """
        Trains and validates for multiple epochs.
        Saves the best model based on validation loss/metrics.
        """
        history = []
        best_val_metric = float('inf') if "pretrain" in mode else 0.0 # Loss for pretrain, Anomaly F1 for finetune
        
        logger.info("Starting %s training for %d epochs...", mode, epochs)
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch, mode)
            val_metrics = self.validate(mode)
            
            # Combine metrics for logging
            epoch_metrics = {**train_metrics, **val_metrics}
            history.append(epoch_metrics)
            
            # Determine logging string
            if "pretrain" in mode:
                logger.info(
                    "Epoch %d/%d | Train Loss: %.4f | Train Token Acc: %.2f%% | Val Loss: %.4f | Val Token Acc: %.2f%% | Val PPL: %.2f",
                    epoch, epochs, epoch_metrics["loss"], epoch_metrics.get("token_accuracy", 0.0)*100,
                    epoch_metrics["val_loss"], epoch_metrics.get("val_token_accuracy", 0.0)*100,
                    epoch_metrics.get("val_perplexity", 0.0)
                )
                
                # Check for checkpoint saving (lowest val loss)
                current_metric = epoch_metrics["val_loss"]
                if current_metric < best_val_metric:
                    best_val_metric = current_metric
                    self.save_checkpoint(f"best_{mode}.pt")
            else:
                logger.info(
                    "Epoch %d/%d | Train Loss: %.4f | Val Anomaly F1: %.2f%% | Val RCA F1 (macro): %.2f%%",
                    epoch, epochs, epoch_metrics["loss"],
                    epoch_metrics.get("val_anomaly_f1", 0.0)*100,
                    epoch_metrics.get("val_rca_f1_macro", 0.0)*100
                )
                
                # Check for checkpoint saving (highest Anomaly F1)
                current_metric = epoch_metrics.get("val_anomaly_f1", 0.0)
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    self.save_checkpoint(f"best_{mode}.pt")
                    
        return history

    def save_checkpoint(self, filename: str):
        """
        Saves model state and weights.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config
        }, path)
        logger.info("Saved checkpoint to %s", path)

    def load_checkpoint(self, filename: str, load_encoder_only: bool = False):
        """
        Loads model weights from checkpoint.
        If load_encoder_only is True, only loads encoder weights (used for pretrain transfer).
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        if load_encoder_only:
            # Filter and load weights matching the encoder prefix only
            model_dict = self.model.state_dict()
            encoder_dict = {
                k: v for k, v in checkpoint["model_state_dict"].items()
                if k.startswith("encoder.")
            }
            # Update current state dict
            model_dict.update(encoder_dict)
            self.model.load_state_dict(model_dict)
            logger.info("Transferred pretrained encoder weights from %s", path)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded full model checkpoint from %s", path)
