import numpy as np
import math
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Try to import sklearn metrics, otherwise fallback to manual implementations
try:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn is not available. Using manual metrics fallback.")

def compute_binary_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes accuracy, precision, recall, f1, and roc-auc for binary anomaly classification.
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    if _SKLEARN_AVAILABLE:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        try:
            # If y_true contains only one class, roc_auc_score throws an error
            auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            auc = 0.5
    else:
        # Manual fallback
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        acc = (tp + tn) / max(1, len(y_true))
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        auc = 0.5 # Manual AUC calculation omitted for simplicity
        
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc": float(auc)
    }

def compute_multiclass_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray) -> Dict[str, float]:
    """
    Computes accuracy and macro precision, recall, f1 for multi-class RCA classification.
    """
    y_pred = np.argmax(y_pred_logits, axis=-1)
    
    if _SKLEARN_AVAILABLE:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
    else:
        # Manual fallback for multi-class accuracy
        acc = np.mean(y_true == y_pred)
        # Macro F1 is computed by taking average of binary F1s
        classes = np.unique(y_true)
        p_list, r_list, f1_list = [], [], []
        for c in range(6): # Assuming 6 RCA classes
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            
            p_c = tp / max(1, tp + fp)
            r_c = tp / max(1, tp + fn)
            f1_c = 2 * p_c * r_c / max(1e-9, p_c + r_c)
            
            p_list.append(p_c)
            r_list.append(r_c)
            f1_list.append(f1_c)
        p = np.mean(p_list)
        r = np.mean(r_list)
        f1 = np.mean(f1_list)
        
    return {
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1)
    }

def compute_language_modeling_metrics(loss: float, correct_preds: int, total_preds: int) -> Dict[str, float]:
    """
    Computes accuracy and perplexity for MLM or Causal next-event prediction.
    """
    accuracy = correct_preds / max(1, total_preds)
    try:
        # Perplexity = exp(CrossEntropyLoss)
        perplexity = math.exp(min(50, loss)) # Clip loss to 50 to prevent overflow
    except OverflowError:
        perplexity = float('inf')
        
    return {
        "accuracy": float(accuracy),
        "perplexity": float(perplexity)
    }
