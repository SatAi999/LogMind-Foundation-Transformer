import torch
from torch.utils.data import Dataset
import random
import logging
from typing import List, Dict, Any, Tuple, Optional
from logmind.dataset.tokenizer import LogTokenizer

logger = logging.getLogger(__name__)

class LogDataset(Dataset):
    """
    PyTorch Dataset for LogMind.
    Supports self-supervised pretraining (MLM and Causal LM) and supervised fine-tuning.
    """
    def __init__(
        self,
        block_sequences: List[List[str]],
        anomaly_labels: List[int],
        rca_labels: List[int],
        tokenizer: LogTokenizer,
        max_len: int = 64,
        mlm_probability: float = 0.15,
        mode: str = "finetune"  # "pretrain_mlm", "pretrain_clm", "finetune"
    ):
        self.block_sequences = block_sequences
        self.anomaly_labels = anomaly_labels
        self.rca_labels = rca_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_probability = mlm_probability
        self.mode = mode

        assert len(self.block_sequences) == len(self.anomaly_labels) == len(self.rca_labels), \
            "All inputs must have the same length."

    def __len__(self) -> int:
        return len(self.block_sequences)

    def set_mode(self, mode: str):
        """
        Dynamically switch dataset mode between pretraining and fine-tuning.
        """
        assert mode in ["pretrain_mlm", "pretrain_clm", "finetune"], f"Invalid mode: {mode}"
        self.mode = mode
        logger.info("Dataset mode set to: %s", mode)

    def _apply_mlm_masking(self, token_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Masked Language Modeling masking rules:
        - 15% of tokens are selected for masking.
        - Of those, 80% are replaced with [MASK].
        - 10% are replaced with a random vocabulary token.
        - 10% are left unchanged.
        - Special tokens ([PAD], [CLS], [SEP]) are not masked.
        - Target labels are set to the original token ID for masked positions, and -100 elsewhere.
        """
        inputs = torch.tensor(token_ids, dtype=torch.long)
        labels = torch.full(inputs.shape, -100, dtype=torch.long)
        
        # Determine positions that can be masked (exclude CLS, SEP, PAD)
        maskable_positions = []
        for idx, token_id in enumerate(token_ids):
            if token_id not in [self.tokenizer.pad_id, self.tokenizer.cls_id, self.tokenizer.sep_id]:
                maskable_positions.append(idx)
                
        if not maskable_positions:
            return inputs, labels
            
        # Select 15% of maskable positions
        num_to_mask = max(1, int(len(maskable_positions) * self.mlm_probability))
        masked_indices = random.sample(maskable_positions, num_to_mask)
        
        for idx in masked_indices:
            # Save original token ID as target
            labels[idx] = inputs[idx]
            
            rand = random.random()
            if rand < 0.80:
                # 80% replaced with [MASK]
                inputs[idx] = self.tokenizer.mask_id
            elif rand < 0.90:
                # 10% replaced with random token from vocabulary (excluding special tokens)
                random_token_id = random.randint(len(self.tokenizer.special_tokens), len(self.tokenizer) - 1)
                inputs[idx] = random_token_id
            else:
                # 10% left unchanged
                pass
                
        return inputs, labels

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sequence = self.block_sequences[index]
        anomaly_label = self.anomaly_labels[index]
        rca_label = self.rca_labels[index]
        
        # Encode sequence
        # We set add_special_tokens=True to add [CLS] and [SEP]
        token_ids = self.tokenizer.encode(sequence, add_special_tokens=True, max_len=self.max_len)
        
        # Construct basic inputs
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        padding_mask = (input_ids != self.tokenizer.pad_id) # True for valid tokens, False for padding
        
        item = {
            "input_ids": input_ids,
            "padding_mask": padding_mask,
            "anomaly_label": torch.tensor(anomaly_label, dtype=torch.float),
            "rca_label": torch.tensor(rca_label, dtype=torch.long)
        }
        
        if self.mode == "pretrain_mlm":
            inputs, labels = self._apply_mlm_masking(token_ids)
            item["input_ids"] = inputs
            item["mlm_labels"] = labels
            
        elif self.mode == "pretrain_clm":
            # Causal Language Modeling (predict next token)
            # Input: T_0, T_1, ..., T_{N-1}
            # Target: T_1, T_2, ..., T_N
            # We construct targets shifted by 1. Padding tokens get target -100.
            inputs = torch.tensor(token_ids, dtype=torch.long)
            labels = torch.full(inputs.shape, -100, dtype=torch.long)
            
            # Find the actual sequence length (excluding padding)
            non_pad_len = padding_mask.sum().item()
            
            # The target for position i is the token at position i+1
            # We exclude [CLS] from predictions (position 0 target is token at position 1)
            # We predict up to the [SEP] token
            for idx in range(0, non_pad_len - 1):
                labels[idx] = inputs[idx + 1]
                
            item["clm_labels"] = labels
            
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function to assemble batches of tensors.
    """
    input_ids = torch.stack([x["input_ids"] for x in batch])
    padding_mask = torch.stack([x["padding_mask"] for x in batch])
    anomaly_label = torch.stack([x["anomaly_label"] for x in batch])
    rca_label = torch.stack([x["rca_label"] for x in batch])
    
    collated = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "anomaly_label": anomaly_label,
        "rca_label": rca_label
    }
    
    if "mlm_labels" in batch[0]:
        collated["mlm_labels"] = torch.stack([x["mlm_labels"] for x in batch])
        
    if "clm_labels" in batch[0]:
        collated["clm_labels"] = torch.stack([x["clm_labels"] for x in batch])
        
    return collated
