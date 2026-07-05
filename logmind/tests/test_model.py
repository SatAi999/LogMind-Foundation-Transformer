import pytest
import torch
from logmind.models.heads import MLMHead, AnomalyHead, RCAHead, ContrastiveHead
from logmind.models.logmind_model import LogMindModel

def test_prediction_heads():
    batch_size = 2
    seq_len = 5
    d_model = 16
    vocab_size = 10
    num_classes_rca = 6
    d_emb = 8
    
    # 1. MLM Head
    h = torch.randn(batch_size, seq_len, d_model)
    mlm_head = MLMHead(d_model, vocab_size)
    mlm_out = mlm_head(h)
    assert mlm_out.shape == (batch_size, seq_len, vocab_size)
    
    # 2. Anomaly Head
    cls_emb = torch.randn(batch_size, d_model)
    anomaly_head = AnomalyHead(d_model, dropout=0.0)
    anomaly_out = anomaly_head(cls_emb)
    assert anomaly_out.shape == (batch_size,)
    
    # 3. RCA Head
    rca_head = RCAHead(d_model, num_classes_rca, dropout=0.0)
    rca_out = rca_head(cls_emb)
    assert rca_out.shape == (batch_size, num_classes_rca)
    
    # 4. Contrastive Head
    contrastive_head = ContrastiveHead(d_model, d_emb)
    emb_out = contrastive_head(cls_emb)
    assert emb_out.shape == (batch_size, d_emb)
    # Check L2 Normalization (norm should be 1.0)
    norms = torch.norm(emb_out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_logmind_model_forward():
    vocab_size = 20
    d_model = 16
    n_heads = 2
    d_ff = 32
    n_layers = 2
    max_seq_len = 10
    
    model = LogMindModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        num_classes_anomaly=1,
        num_classes_rca=6,
        d_emb=8,
        embed_type="learned",
        norm_type="pre",
        dropout=0.1
    )
    
    batch_size = 2
    seq_len = 6
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Mark last 2 tokens of sequence index 0 as padding, and sequence index 1 as fully valid
    padding_mask = torch.tensor([
        [True, True, True, True, False, False],
        [True, True, True, True, True, True]
    ])
    
    # Test Bidirectional pass
    outputs_bidir = model(input_ids, padding_mask, mask_type="bidirectional")
    
    assert "mlm_logits" in outputs_bidir
    assert "anomaly_logits" in outputs_bidir
    assert "rca_logits" in outputs_bidir
    assert "embeddings" in outputs_bidir
    assert "attention_maps" in outputs_bidir
    
    assert outputs_bidir["mlm_logits"].shape == (batch_size, seq_len, vocab_size)
    assert outputs_bidir["anomaly_logits"].shape == (batch_size,)
    assert outputs_bidir["rca_logits"].shape == (batch_size, 6)
    assert outputs_bidir["embeddings"].shape == (batch_size, 8)
    
    # Test Causal pass
    outputs_causal = model(input_ids, padding_mask, mask_type="causal")
    assert outputs_causal["mlm_logits"].shape == (batch_size, seq_len, vocab_size)
    
    # Test Selective Heads pass
    outputs_selective = model(input_ids, padding_mask, run_heads=["anomaly", "rca"])
    assert "anomaly_logits" in outputs_selective
    assert "rca_logits" in outputs_selective
    assert "mlm_logits" not in outputs_selective
    assert "embeddings" not in outputs_selective

def test_attention_mask_types():
    vocab_size = 10
    model = LogMindModel(vocab_size=vocab_size, d_model=8, n_heads=1, n_layers=1)
    
    # Sequence of length 4
    # Sequence 0: 3 valid tokens, 1 pad token
    # Sequence 1: 4 valid tokens
    padding_mask = torch.tensor([
        [True, True, True, False],
        [True, True, True, True]
    ])
    
    # 1. Bidirectional mask verification
    bidir_mask = model.get_attention_mask(padding_mask, mask_type="bidirectional")
    # Shape should be (batch_size, 1, seq_len, seq_len)
    assert bidir_mask.shape == (2, 1, 4, 4)
    
    # In sequence 0, column index 3 (corresponds to padding key) must be masked (-1e9) for all query positions
    # All other columns should be 0.0 (active)
    assert torch.all(bidir_mask[0, 0, :, 3] == -1e9)
    assert torch.all(bidir_mask[0, 0, :, :3] == 0.0)
    
    # In sequence 1, all elements should be 0.0 (no padding)
    assert torch.all(bidir_mask[1] == 0.0)
    
    # 2. Causal mask verification
    causal_mask = model.get_attention_mask(padding_mask, mask_type="causal")
    # Query position i cannot attend to key position j > i
    # Let's inspect sequence 1 (fully valid)
    for q_idx in range(4):
        for k_idx in range(4):
            val = causal_mask[1, 0, q_idx, k_idx].item()
            if k_idx > q_idx:
                assert val == -1e9  # Masked future
            else:
                assert val == 0.0   # Allowed past
                
    # Inspect sequence 0 (3 valid tokens, 1 pad token)
    # Token 3 is padding. Query position 2 (valid) should be able to attend to index 0, 1, 2, but not index 3
    assert torch.all(causal_mask[0, 0, 2] == torch.tensor([0.0, 0.0, 0.0, -1e9], device=padding_mask.device))
