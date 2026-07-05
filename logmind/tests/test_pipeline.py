import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from logmind.dataset.tokenizer import LogTokenizer
from logmind.dataset.dataset import LogDataset, collate_fn
from logmind.models.logmind_model import LogMindModel
from logmind.training.losses import PairwiseContrastiveLoss, LogMindLoss
from logmind.training.trainer import LogMindTrainer
from logmind.training.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_language_modeling_metrics
)

def test_pairwise_contrastive_loss():
    loss_fn = PairwiseContrastiveLoss(margin=0.5)
    
    # 3 embeddings of size 4 (L2 normalized)
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],  # Exactly identical (similarity 1.0)
        [0.0, 1.0, 0.0, 0.0]   # Orthogonal (similarity 0.0)
    ])
    
    # Target categories: elements 0 and 1 are in same category, element 2 is different
    labels = torch.tensor([0, 0, 1])
    
    loss = loss_fn(embeddings, labels)
    
    # Pos pair: index (0,1) and (1,0) similarity = 1.0. Loss pos = 1.0 - 1.0 = 0.0.
    # Neg pairs: index (0,2), (2,0), (1,2), (2,1) similarity = 0.0. Margin = 0.5. Loss neg = max(0, 0.0 - 0.5) = 0.0.
    # Total loss should be close to 0.0
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)
    
    # Now verify with negative pairs that violate margin
    embeddings_bad = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.6, 0.0, 0.0]   # Different class but close (similarity 0.8)
    ])
    labels_bad = torch.tensor([0, 1])
    
    loss_bad = loss_fn(embeddings_bad, labels_bad)
    # Cos similarity = 0.8. Margin = 0.5. neg loss = max(0, 0.8 - 0.5) = 0.3.
    # Total loss should be 0.3
    assert torch.allclose(loss_bad, torch.tensor(0.3), atol=1e-5)

def test_metrics_computations():
    # 1. Binary metrics
    y_true_bin = np.array([0, 1, 0, 1])
    y_pred_bin = np.array([0.1, 0.9, 0.2, 0.8])
    metrics_bin = compute_binary_metrics(y_true_bin, y_pred_bin)
    assert metrics_bin["accuracy"] == 1.0
    assert metrics_bin["f1"] == 1.0
    assert metrics_bin["auc"] == 1.0
    
    # 2. Multiclass metrics
    y_true_multi = np.array([0, 1, 2, 0])
    y_pred_multi = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [10.0, 0.0, 0.0]
    ])
    metrics_multi = compute_multiclass_metrics(y_true_multi, y_pred_multi)
    assert metrics_multi["accuracy"] == 1.0
    assert metrics_multi["f1_macro"] == 1.0
    
    # 3. Language modeling metrics
    metrics_lm = compute_language_modeling_metrics(0.0, 8, 10)
    assert metrics_lm["accuracy"] == 0.8
    assert metrics_lm["perplexity"] == 1.0

def test_end_to_end_training_mini_pipeline(tmp_path):
    device = torch.device("cpu")
    
    # Setup dummy vocabulary and tokenizer
    tokenizer = LogTokenizer()
    templates = [f"Template_{i}" for i in range(10)]
    tokenizer.build_vocab(templates)
    
    # Setup dummy dataset
    sequences = [["Template_0", "Template_1", "Template_2", "Template_0"] for _ in range(8)]
    anomaly_labels = [0, 1, 0, 1, 0, 1, 0, 1]
    rca_labels = [0, 1, 0, 2, 0, 3, 0, 4]
    
    dataset = LogDataset(
        sequences, anomaly_labels, rca_labels, tokenizer, 
        max_len=8, mlm_probability=0.15, mode="pretrain_mlm"
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Miniature model
    model = LogMindModel(
        vocab_size=len(tokenizer),
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        max_seq_len=8,
        num_classes_anomaly=1,
        num_classes_rca=6,
        d_emb=4
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = LogMindLoss()
    
    # Trainer
    trainer = LogMindTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config={},
        checkpoint_dir=str(tmp_path)
    )
    
    # 1. Test MLM pretraining epoch
    dataset.set_mode("pretrain_mlm")
    mlm_metrics = trainer.train_epoch(1, mode="pretrain_mlm")
    assert "mlm_loss" in mlm_metrics
    assert mlm_metrics["loss"] > 0
    assert "token_accuracy" in mlm_metrics
    
    # 2. Test CLM pretraining epoch
    dataset.set_mode("pretrain_clm")
    clm_metrics = trainer.train_epoch(1, mode="pretrain_clm")
    assert "clm_loss" in clm_metrics
    assert clm_metrics["loss"] > 0
    assert "token_accuracy" in clm_metrics
    
    # 3. Test Fine-tuning epoch
    dataset.set_mode("finetune")
    ft_metrics = trainer.train_epoch(1, mode="finetune")
    assert "anomaly_loss" in ft_metrics
    assert "rca_loss" in ft_metrics
    assert "contrastive_loss" in ft_metrics
    assert ft_metrics["loss"] > 0
    
    # 4. Test validation
    val_metrics = trainer.validate("finetune")
    assert "val_anomaly_f1" in val_metrics
    assert "val_rca_f1_macro" in val_metrics
    assert "val_loss" in val_metrics
    
    # 5. Test Checkpoint saving and loading
    trainer.save_checkpoint("test_ckpt.pt")
    
    # Load full checkpoint
    new_model = LogMindModel(
        vocab_size=len(tokenizer),
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        max_seq_len=8,
        num_classes_anomaly=1,
        num_classes_rca=6,
        d_emb=4
    ).to(device)
    
    new_trainer = LogMindTrainer(
        model=new_model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config={},
        checkpoint_dir=str(tmp_path)
    )
    
    new_trainer.load_checkpoint("test_ckpt.pt")
    
    # Load encoder only (transfer learning check)
    transfer_model = LogMindModel(
        vocab_size=len(tokenizer),
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        max_seq_len=8,
        num_classes_anomaly=1,
        num_classes_rca=6,
        d_emb=4
    ).to(device)
    
    transfer_trainer = LogMindTrainer(
        model=transfer_model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config={},
        checkpoint_dir=str(tmp_path)
    )
    
    transfer_trainer.load_checkpoint("test_ckpt.pt", load_encoder_only=True)
