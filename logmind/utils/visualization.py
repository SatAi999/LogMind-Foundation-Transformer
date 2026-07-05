import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Ensure styling is modern and clean
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

def plot_attention_heatmap(
    attention_matrix: np.ndarray, 
    tokens: List[str], 
    save_path: str,
    title: str = "Attention Map",
    head_idx: int = 0
):
    """
    Plots a heatmap for a single attention head.
    Args:
        attention_matrix: Attention weights of shape (num_heads, seq_len, seq_len)
        tokens: Token strings of length seq_len
        save_path: Path to save the image
        title: Title of the plot
        head_idx: Index of the attention head to plot
    """
    # Select specific head: shape (seq_len, seq_len)
    attn = attention_matrix[head_idx]
    seq_len = len(tokens)
    attn = attn[:seq_len, :seq_len]
    
    plt.figure(figsize=(10, 8))
    # Using a vibrant modern colormap (rocket/magma/viridis)
    sns.heatmap(
        attn, 
        xticklabels=tokens, 
        yticklabels=tokens, 
        annot=True, 
        fmt=".2f", 
        cmap="viridis",
        cbar_kws={'label': 'Attention Weight'}
    )
    plt.title(f"{title} (Head {head_idx})")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved attention heatmap to %s", save_path)


def plot_embeddings_tsne(
    embeddings: np.ndarray, 
    labels: np.ndarray, 
    save_path: str,
    title: str = "t-SNE Log Sequence Embeddings"
):
    """
    Visualizes high-dimensional log embeddings in 2D using t-SNE.
    Colors points by their RCA root cause category.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.error("scikit-learn is required for t-SNE visualization.")
        return

    perp = min(30, max(5, len(embeddings) // 10))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    rca_categories = {
        0: "Normal",
        1: "WriteFailure",
        2: "ConnectionTimeout",
        3: "ServingFailure",
        4: "ReplicaVolumeError",
        5: "OtherAnomaly"
    }
    
    label_names = [rca_categories.get(int(l), f"Class {l}") for l in labels]

    plt.figure(figsize=(10, 8))
    # Curated premium color palette
    palette = sns.color_palette("bright", len(np.unique(labels)))
    sns.scatterplot(
        x=emb_2d[:, 0], 
        y=emb_2d[:, 1], 
        hue=label_names, 
        palette="Set2",
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5,
        s=40
    )
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Root Cause Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved t-SNE embedding plot to %s", save_path)


def plot_training_curves(history: List[Dict[str, float]], save_path: str, mode: str):
    """
    Plots training and validation loss curves.
    """
    epochs = range(1, len(history) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Curve Plot
    plt.subplot(1, 2, 1)
    train_loss = [h["loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    plt.plot(epochs, train_loss, 'o-', label="Train Loss", color="#4C72B0")
    plt.plot(epochs, val_loss, 's-', label="Val Loss", color="#C44E52")
    plt.title(f"{mode.capitalize()} Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # Metric Curve Plot
    plt.subplot(1, 2, 2)
    if "pretrain" in mode:
        train_acc = [h.get("token_accuracy", 0.0) * 100 for h in history]
        val_acc = [h.get("val_token_accuracy", 0.0) * 100 for h in history]
        plt.plot(epochs, train_acc, 'o-', label="Train Token Acc", color="#55A868")
        plt.plot(epochs, val_acc, 's-', label="Val Token Acc", color="#937860")
        plt.title("Pretrain Token Accuracy (%)")
        plt.ylabel("Accuracy (%)")
    else:
        val_f1_anom = [h.get("val_anomaly_f1", 0.0) * 100 for h in history]
        val_f1_rca = [h.get("val_rca_f1_macro", 0.0) * 100 for h in history]
        plt.plot(epochs, val_f1_anom, 'o-', label="Val Anomaly F1", color="#CCB974")
        plt.plot(epochs, val_f1_rca, 's-', label="Val RCA Macro F1", color="#64B5CD")
        plt.title("Fine-tuning F1 Scores (%)")
        plt.ylabel("F1 Score (%)")
        
    plt.xlabel("Epochs")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved training curves to %s", save_path)


def plot_anomaly_timeline(
    anomaly_scores: List[float], 
    labels: List[int], 
    save_path: str,
    title: str = "Log Sequence Anomaly Timeline"
):
    """
    Plots an anomaly timeline showing prediction confidence scores for sequential inputs.
    Red circles represent actual anomalies.
    """
    times = range(len(anomaly_scores))
    
    plt.figure(figsize=(12, 5))
    plt.plot(times, anomaly_scores, label="Anomaly Probability", color="#4C72B0", alpha=0.6, linewidth=1.5)
    
    # Add a horizontal threshold line
    plt.axhline(y=0.5, color='orange', linestyle='--', label="Anomaly Threshold (0.5)")
    
    # Highlight actual anomaly points
    anom_indices = [i for i, l in enumerate(labels) if l == 1]
    anom_scores = [anomaly_scores[i] for i in anom_indices]
    
    plt.scatter(anom_indices, anom_scores, color="red", label="True Anomaly Events", s=30, zorder=5)
    
    plt.title(title)
    plt.xlabel("Log Sequences (Time Index)")
    plt.ylabel("Anomaly Score / Confidence")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved anomaly timeline plot to %s", save_path)
