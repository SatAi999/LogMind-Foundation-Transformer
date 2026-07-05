import os
import yaml
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from logmind.dataset.parser import LogParser
from logmind.dataset.tokenizer import LogTokenizer
from logmind.models.logmind_model import LogMindModel

logger = logging.getLogger(__name__)

class LogMindInference:
    """
    High-level API for model inference, autoregressive generation,
    and incident similarity search.
    """
    def __init__(self, checkpoint_path: str = "checkpoints/best_finetune.pt", 
                 vocab_path: str = "checkpoints/vocab.json", 
                 config_path: str = "logmind/config/config.yaml"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load tokenizer
        self.tokenizer = LogTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        
        # Load parser
        self.parser = LogParser()
        
        # Load model
        self.model = LogMindModel(
            vocab_size=len(self.tokenizer),
            d_model=self.config["model"]["d_model"],
            n_heads=self.config["model"]["n_heads"],
            d_ff=self.config["model"]["d_ff"],
            n_layers=self.config["model"]["n_layers"],
            max_seq_len=self.config["data"]["max_len"],
            num_classes_anomaly=self.config["model"]["num_classes_anomaly"],
            num_classes_rca=self.config["model"]["num_classes_rca"],
            embed_type=self.config["model"]["embed_type"]
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("Loaded LogMind inference model from %s on device: %s", checkpoint_path, self.device)
        
        # Incident Similarity Search database variables
        self.indexed_sequences: List[List[str]] = []
        self.indexed_embeddings: Optional[np.ndarray] = None
        self.indexed_rca_labels: List[int] = []

    def build_similarity_database(self, log_path: str, label_path: str, max_lines: int = 200000):
        """
        Parses a subset of logs and computes their embeddings to form the similarity search database.
        """
        logger.info("Indexing logs for similarity search...")
        block_sequences, _, block_rca = self.parser.parse_file(log_path, label_path, max_lines=max_lines)
        
        indexed_seqs = []
        indexed_rcas = []
        
        # Filter sequences to keep unique ones or a representative sample
        for bid, seq in block_sequences.items():
            if len(seq) > 2: # Keep non-trivial sequences
                indexed_seqs.append(seq)
                indexed_rcas.append(block_rca[bid])
                
        # Batch embed
        all_embs = []
        batch_size = 128
        
        for i in range(0, len(indexed_seqs), batch_size):
            batch_seqs = indexed_seqs[i:i+batch_size]
            
            token_ids_list = [self.tokenizer.encode(seq, add_special_tokens=True, max_len=self.config["data"]["max_len"]) for seq in batch_seqs]
            
            input_tensor = torch.tensor(token_ids_list, dtype=torch.long).to(self.device)
            padding_mask = (input_tensor != self.tokenizer.pad_id).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor, padding_mask, mask_type="bidirectional", run_heads=["contrastive"])
                embs = outputs["embeddings"].cpu().numpy()
                all_embs.append(embs)
                
        if all_embs:
            self.indexed_embeddings = np.vstack(all_embs)
            self.indexed_sequences = indexed_seqs
            self.indexed_rca_labels = indexed_rcas
            logger.info("Similarity database built. Indexed %d sequences.", len(self.indexed_sequences))
        else:
            logger.warning("No sequences found to index.")

    def predict_sequence(self, raw_log_lines: List[str]) -> Dict[str, Any]:
        """
        Parses, tokenizes, and runs multi-task inference on a sequence of raw log lines.
        """
        # Parse templates
        templates = []
        for line in raw_log_lines:
            parsed = self.parser.parse_line(line)
            msg = parsed["message"] if parsed else line.strip()
            templates.append(self.parser.get_template(msg))
            
        token_ids = self.tokenizer.encode(templates, add_special_tokens=True, max_len=self.config["data"]["max_len"])
        
        # Actual valid length (excluding padding)
        valid_len = sum(1 for idx in token_ids if idx != self.tokenizer.pad_id)
        valid_tokens = self.tokenizer.decode(token_ids[:valid_len], skip_special_tokens=False)
        
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        padding_mask = (input_tensor != self.tokenizer.pad_id).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor, padding_mask, mask_type="bidirectional")
            
            anomaly_prob = torch.sigmoid(outputs["anomaly_logits"])[0].item()
            rca_logits = outputs["rca_logits"][0]
            rca_probs = torch.softmax(rca_logits, dim=-1).cpu().numpy()
            
            embedding = outputs["embeddings"][0].cpu().numpy()
            
            # Extract attention map from the final block: shape (num_heads, seq_len, seq_len)
            attention_map = outputs["attention_maps"][-1][0].cpu().numpy()
            
        rca_categories = {
            0: "Normal",
            1: "WriteFailure",
            2: "ConnectionTimeout",
            3: "ServingFailure",
            4: "ReplicaVolumeError",
            5: "OtherAnomaly"
        }
        
        rca_probs_dict = {rca_categories[i]: float(rca_probs[i]) for i in range(len(rca_probs))}
        predicted_rca = rca_categories[np.argmax(rca_probs)]
        
        return {
            "templates": templates,
            "valid_tokens": valid_tokens,
            "anomaly_probability": anomaly_prob,
            "predicted_rca": predicted_rca,
            "rca_probabilities": rca_probs_dict,
            "embedding": embedding,
            "attention_map": attention_map
        }

    def generate_next_events(self, prompt_templates: List[str], max_gen_len: int = 10, temperature: float = 1.0) -> List[str]:
        """
        Autoregressively generates next log event templates using the causal mask.
        """
        # Encode initial prompt
        # We start with [CLS] and add prompt templates
        # We DO NOT pad yet, we will pad dynamically or handle it sequence-by-sequence
        current_seq = prompt_templates.copy()
        
        for _ in range(max_gen_len):
            token_ids = self.tokenizer.encode(current_seq, add_special_tokens=True, max_len=None)
            
            # Strip trailing [SEP] during generation loop because we append tokens,
            # and [SEP] should only appear when the model generates it or we finish.
            if token_ids[-1] == self.tokenizer.sep_id:
                token_ids = token_ids[:-1]
                
            input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
            padding_mask = torch.ones_like(input_tensor, dtype=torch.bool).to(self.device)
            
            with torch.no_grad():
                # For generation, we MUST use CAUSAL mask!
                outputs = self.model(input_tensor, padding_mask, mask_type="causal", run_heads=["mlm"])
                
                # Get logits for the LAST token position
                last_logits = outputs["mlm_logits"][0, -1, :]
                
                if temperature == 0.0:
                    # Greedy search
                    next_token_id = torch.argmax(last_logits).item()
                else:
                    # Sample search
                    probs = torch.softmax(last_logits / temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                    
            next_token = self.tokenizer.id2token.get(next_token_id, self.tokenizer.unk_token)
            
            # Stop generation if model predicts [SEP]
            if next_token_id == self.tokenizer.sep_id:
                break
                
            # If it's a special token that isn't SEP, skip it to prevent junk generation
            if next_token in self.tokenizer.special_tokens:
                continue
                
            current_seq.append(next_token)
            
        return current_seq

    def search_similar_incidents(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search against the indexed database of log incidents.
        """
        if self.indexed_embeddings is None:
            raise ValueError("Similarity database has not been built yet. Call build_similarity_database().")
            
        # Compute cosine similarity (dot product of normalized embeddings)
        # query_embedding: (d_emb,)
        # indexed_embeddings: (N, d_emb)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(self.indexed_embeddings, query_norm)
        
        # Sort and get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        rca_categories = {
            0: "Normal",
            1: "WriteFailure",
            2: "ConnectionTimeout",
            3: "ServingFailure",
            4: "ReplicaVolumeError",
            5: "OtherAnomaly"
        }
        
        results = []
        for idx in top_indices:
            results.append({
                "sequence": self.indexed_sequences[idx],
                "similarity": float(similarities[idx]),
                "rca_category": rca_categories.get(self.indexed_rca_labels[idx], "OtherAnomaly")
            })
            
        return results
