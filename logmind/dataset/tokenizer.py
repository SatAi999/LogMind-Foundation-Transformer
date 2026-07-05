import json
import os
import logging
from typing import List, Dict, Union, Optional

logger = logging.getLogger(__name__)

class LogTokenizer:
    """
    Tokenizer for log event templates.
    Maps sequences of templates to token ID sequences.
    """
    def __init__(self, pad_token: str = "[PAD]", unk_token: str = "[UNK]", 
                 cls_token: str = "[CLS]", sep_token: str = "[SEP]", 
                 mask_token: str = "[MASK]"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        
        self.special_tokens = [pad_token, unk_token, cls_token, sep_token, mask_token]
        
        # Token to ID mapping
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        
        # Initialize with special tokens
        for token in self.special_tokens:
            self._add_token(token)

    @property
    def pad_id(self) -> int:
        return self.token2id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token2id[self.unk_token]

    @property
    def cls_id(self) -> int:
        return self.token2id[self.cls_token]

    @property
    def sep_id(self) -> int:
        return self.token2id[self.sep_token]

    @property
    def mask_id(self) -> int:
        return self.token2id[self.mask_token]

    def _add_token(self, token: str):
        if token not in self.token2id:
            new_id = len(self.token2id)
            self.token2id[token] = new_id
            self.id2token[new_id] = token

    def build_vocab(self, templates: List[str]):
        """
        Builds the vocabulary from a list of template strings.
        """
        # Sort templates for deterministic vocabulary building
        for template in sorted(set(templates)):
            if template not in self.token2id:
                self._add_token(template)
        logger.info("Vocabulary built with %d tokens (including special tokens).", len(self))

    def __len__(self) -> int:
        return len(self.token2id)

    def encode(self, sequence: List[str], add_special_tokens: bool = True, max_len: Optional[int] = None) -> List[int]:
        """
        Encodes a sequence of log templates into a list of token IDs.
        """
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.cls_id)
            
        for token in sequence:
            token_ids.append(self.token2id.get(token, self.unk_id))
            
        if add_special_tokens:
            token_ids.append(self.sep_id)
            
        if max_len is not None:
            if len(token_ids) > max_len:
                # Truncate keeping CLS and SEP if present
                if add_special_tokens:
                    token_ids = token_ids[:max_len - 1] + [self.sep_id]
                else:
                    token_ids = token_ids[:max_len]
            else:
                # Pad sequence
                padding = [self.pad_id] * (max_len - len(token_ids))
                token_ids.extend(padding)
                
        return token_ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]:
        """
        Decodes a list of token IDs back into template strings.
        """
        tokens = []
        for i in ids:
            token = self.id2token.get(i, self.unk_token)
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return tokens

    def save_vocab(self, filepath: str):
        """
        Saves vocabulary map to file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"token2id": self.token2id, "id2token": {str(k): v for k, v in self.id2token.items()}}, f, indent=4)
        logger.info("Vocabulary saved to %s", filepath)

    def load_vocab(self, filepath: str):
        """
        Loads vocabulary map from file.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = {int(k): v for k, v in data["id2token"].items()}
        logger.info("Vocabulary loaded from %s. Size: %d", filepath, len(self))
