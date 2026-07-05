import os
import yaml
import torch
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from typing import Dict, Any, List

from logmind.dataset.parser import LogParser
from logmind.dataset.tokenizer import LogTokenizer
from logmind.dataset.dataset import LogDataset, collate_fn
from logmind.models.logmind_model import LogMindModel
from logmind.training.losses import LogMindLoss
from logmind.training.trainer import LogMindTrainer
from logmind.training.metrics import compute_binary_metrics, compute_multiclass_metrics
from logmind.utils.visualization import (
    plot_training_curves, 
    plot_embeddings_tsne, 
    plot_anomaly_timeline,
    plot_attention_heatmap
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LogMindMain")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # 1. Load config
    config_path = "logmind/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config["data"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() and config["training"]["device"] == "cuda" else "cpu")
    logger.info("Using device: %s", device)
    
    # 2. Parse logs
    log_path = config["data"]["log_path"]
    label_path = config["data"]["label_path"]
    max_lines = config["data"]["max_lines"]
    
    parser = LogParser()
    block_sequences, block_labels, block_rca = parser.parse_file(
        log_path, label_path, max_lines=max_lines
    )
    
    # 3. Build vocab and tokenizer
    all_templates = []
    for seq in block_sequences.values():
        all_templates.extend(seq)
        
    tokenizer = LogTokenizer()
    tokenizer.build_vocab(all_templates)
    
    # Save vocab
    vocab_path = config["data"]["vocab_path"]
    tokenizer.save_vocab(vocab_path)
    
    # 4. Prepare data splits
    block_ids = list(block_sequences.keys())
    random.shuffle(block_ids)
    
    total_blocks = len(block_ids)
    train_size = int(total_blocks * config["data"]["train_split"])
    val_size = int(total_blocks * config["data"]["val_split"])
    
    train_ids = block_ids[:train_size]
    val_ids = block_ids[train_size:train_size + val_size]
    test_ids = block_ids[train_size + val_size:]
    
    logger.info("Dataset splits: Train=%d, Val=%d, Test=%d", len(train_ids), len(val_ids), len(test_ids))
    
    def get_split_data(ids_list):
        seqs = [block_sequences[bid] for bid in ids_list]
        lbls = [block_labels[bid] for bid in ids_list]
        rcas = [block_rca[bid] for bid in ids_list]
        return seqs, lbls, rcas
        
    train_seqs, train_lbls, train_rcas = get_split_data(train_ids)
    val_seqs, val_lbls, val_rcas = get_split_data(val_ids)
    test_seqs, test_lbls, test_rcas = get_split_data(test_ids)
    
    # 5. Initialize model
    model = LogMindModel(
        vocab_size=len(tokenizer),
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        n_layers=config["model"]["n_layers"],
        max_seq_len=config["data"]["max_len"],
        num_classes_anomaly=config["model"]["num_classes_anomaly"],
        num_classes_rca=config["model"]["num_classes_rca"],
        embed_type=config["model"]["embed_type"],
        dropout=config["model"]["dropout"]
    ).to(device)
    
    logger.info("Initialized LogMind Transformer with parameters: d_model=%d, heads=%d, layers=%d", 
                config["model"]["d_model"], config["model"]["n_heads"], config["model"]["n_layers"])
    
    # 6. STAGE 1: Self-supervised pretraining (MLM)
    pretrain_cfg = config["training"]["pretrain"]
    train_dataset = LogDataset(
        train_seqs, train_lbls, train_rcas, tokenizer, 
        max_len=config["data"]["max_len"], 
        mlm_probability=config["data"]["mlm_probability"],
        mode="pretrain_mlm"
    )
    val_dataset = LogDataset(
        val_seqs, val_lbls, val_rcas, tokenizer, 
        max_len=config["data"]["max_len"], 
        mode="pretrain_mlm"
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=pretrain_cfg["batch_size"], 
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=pretrain_cfg["batch_size"], 
        shuffle=False, collate_fn=collate_fn
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=pretrain_cfg["learning_rate"], 
        weight_decay=pretrain_cfg["weight_decay"]
    )
    loss_fn = LogMindLoss()
    
    trainer = LogMindTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config,
        checkpoint_dir="checkpoints"
    )
    
    logger.info("=== Starting Stage 1: Self-Supervised MLM Pretraining ===")
    pretrain_history = trainer.fit(pretrain_cfg["epochs"], mode="pretrain_mlm")
    
    # Plot pretrain curves
    plot_training_curves(pretrain_history, "plots/pretraining_curves.png", "pretrain_mlm")
    
    # 7. STAGE 2: Supervised Fine-Tuning
    finetune_cfg = config["training"]["finetune"]
    
    # Set dataset modes to fine-tuning
    train_dataset.set_mode("finetune")
    val_dataset.set_mode("finetune")
    
    # Re-initialize trainer dataloaders for fine-tuning
    trainer.train_loader = DataLoader(
        train_dataset, batch_size=finetune_cfg["batch_size"], 
        shuffle=True, collate_fn=collate_fn
    )
    trainer.val_loader = DataLoader(
        val_dataset, batch_size=finetune_cfg["batch_size"], 
        shuffle=False, collate_fn=collate_fn
    )
    
    # Load best MLM weights back into model
    trainer.load_checkpoint("best_pretrain_mlm.pt", load_encoder_only=True)
    
    # Set up optimizer for fine-tuning (often lower LR)
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=finetune_cfg["learning_rate"], 
        weight_decay=finetune_cfg["weight_decay"]
    )
    
    logger.info("=== Starting Stage 2: Supervised Fine-Tuning (Anomaly + RCA + Similarity) ===")
    finetune_history = trainer.fit(finetune_cfg["epochs"], mode="finetune")
    
    # Plot finetune curves
    plot_training_curves(finetune_history, "plots/finetuning_curves.png", "finetune")
    
    # 8. Evaluation on Test Set
    logger.info("=== Evaluating Model on Test Split ===")
    trainer.load_checkpoint("best_finetune.pt")
    
    test_dataset = LogDataset(
        test_seqs, test_lbls, test_rcas, tokenizer, 
        max_len=config["data"]["max_len"], 
        mode="finetune"
    )
    test_loader = DataLoader(
        test_dataset, batch_size=finetune_cfg["batch_size"], 
        shuffle=False, collate_fn=collate_fn
    )
    
    model.eval()
    all_anomaly_true = []
    all_anomaly_pred = []
    all_rca_true = []
    all_rca_pred = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input_ids"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            outputs = model(inputs, padding_mask, mask_type="bidirectional", run_heads=["anomaly", "rca", "contrastive"])
            
            anomaly_probs = torch.sigmoid(outputs["anomaly_logits"])
            all_anomaly_true.extend(batch["anomaly_label"].cpu().numpy())
            all_anomaly_pred.extend(anomaly_probs.cpu().numpy())
            
            rca_logits = outputs["rca_logits"]
            all_rca_true.extend(batch["rca_label"].cpu().numpy())
            all_rca_pred.extend(rca_logits.cpu().numpy())
            
            all_embeddings.extend(outputs["embeddings"].cpu().numpy())
            
    # Calculate scores on test split
    test_anom_metrics = compute_binary_metrics(
        np.array(all_anomaly_true), 
        np.array(all_anomaly_pred)
    )
    test_rca_metrics = compute_multiclass_metrics(
        np.array(all_rca_true), 
        np.array(all_rca_pred)
    )
    
    logger.info("=== FINAL TEST METRICS ===")
    logger.info("Failure Prediction Accuracy : %.4f", test_anom_metrics["accuracy"])
    logger.info("Failure Prediction Precision: %.4f", test_anom_metrics["precision"])
    logger.info("Failure Prediction Recall   : %.4f", test_anom_metrics["recall"])
    logger.info("Failure Prediction F1 Score : %.4f", test_anom_metrics["f1"])
    logger.info("Failure Prediction ROC-AUC  : %.4f", test_anom_metrics["auc"])
    
    logger.info("RCA Classification Accuracy : %.4f", test_rca_metrics["accuracy"])
    logger.info("RCA Classification Macro F1 : %.4f", test_rca_metrics["f1_macro"])
    
    # Save final test results report
    import json
    report = {
        "failure_prediction": test_anom_metrics,
        "root_cause_analysis": test_rca_metrics
    }
    with open("checkpoints/test_metrics.json", "w") as f:
        json.dump(report, f, indent=4)

    # 9. Generate and save visualizations
    
    # A. Plot t-SNE of test embeddings (subset of 1000 for clarity)
    subset_size = min(1000, len(all_embeddings))
    plot_embeddings_tsne(
        np.array(all_embeddings[:subset_size]), 
        np.array(all_rca_true[:subset_size]), 
        "plots/test_embeddings_tsne.png"
    )
    
    # B. Plot anomaly timeline (subset of 100 points)
    timeline_size = min(100, len(all_anomaly_pred))
    plot_anomaly_timeline(
        all_anomaly_pred[:timeline_size], 
        all_anomaly_true[:timeline_size], 
        "plots/anomaly_timeline.png"
    )
    
    # C. Plot attention heatmap on a sample failure sequence
    # Find a sequence that has an anomaly label
    sample_idx = -1
    for idx, l in enumerate(test_lbls):
        if l == 1 and len(test_seqs[idx]) > 3:
            sample_idx = idx
            break
            
    if sample_idx != -1:
        sample_seq = test_seqs[sample_idx]
        sample_token_ids = tokenizer.encode(sample_seq, add_special_tokens=True, max_len=config["data"]["max_len"])
        
        # Strip padding tokens to get actual tokens to show in plot labels
        valid_len = sum(1 for idx in sample_token_ids if idx != tokenizer.pad_id)
        valid_token_ids = sample_token_ids[:valid_len]
        valid_tokens = tokenizer.decode(valid_token_ids, skip_special_tokens=False)
        
        input_tensor = torch.tensor([sample_token_ids]).to(device)
        pad_mask = torch.tensor([[idx != tokenizer.pad_id for idx in sample_token_ids]]).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor, pad_mask, mask_type="bidirectional")
            # Get attention maps from the last block
            # Shape is (batch_size, num_heads, seq_len, seq_len) -> (1, H, L, L)
            attn_map = outputs["attention_maps"][-1][0].cpu().numpy()
            
        plot_attention_heatmap(
            attn_map, 
            valid_tokens, 
            "plots/sample_attention_heatmap.png",
            title="Attention Heatmap of Anomaly Sequence"
        )
        
    logger.info("All training, evaluation, and visualizations completed successfully. Saved models to 'checkpoints/' and plots to 'plots/'.")

if __name__ == "__main__":
    main()
