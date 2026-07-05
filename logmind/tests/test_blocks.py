import pytest
import torch
import torch.nn as nn
from logmind.models.blocks import LayerNorm, FeedForwardNetwork, ResidualConnection
from logmind.models.encoder import TransformerEncoderBlock, TransformerEncoder

def test_layer_norm_equivalence():
    batch_size = 2
    seq_len = 5
    d_model = 16
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize manual and native LayerNorms
    manual_ln = LayerNorm(d_model)
    native_ln = nn.LayerNorm(d_model)
    
    # Copy parameters to compare direct math
    with torch.no_grad():
        manual_ln.gamma.copy_(native_ln.weight)
        manual_ln.beta.copy_(native_ln.bias)
        
    out_manual = manual_ln(x)
    out_native = native_ln(x)
    
    # Assert mathematical equivalence
    assert torch.allclose(out_manual, out_native, atol=1e-5)
    
    # Verify mean and variance along last dim are close to 0 and 1 before gamma/beta
    # If gamma=1 and beta=0, mean should be ~0, var should be ~1
    raw_ln = LayerNorm(d_model)
    out_raw = raw_ln(x)
    assert torch.allclose(out_raw.mean(dim=-1), torch.zeros(batch_size, seq_len), atol=1e-5)
    assert torch.allclose(out_raw.var(dim=-1, unbiased=False), torch.ones(batch_size, seq_len), atol=1e-4)

def test_feed_forward_network():
    batch_size = 2
    seq_len = 6
    d_model = 16
    d_ff = 64
    
    ffn_gelu = FeedForwardNetwork(d_model, d_ff, activation="gelu")
    ffn_relu = FeedForwardNetwork(d_model, d_ff, activation="relu")
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    out_gelu = ffn_gelu(x)
    out_relu = ffn_relu(x)
    
    assert out_gelu.shape == (batch_size, seq_len, d_model)
    assert out_relu.shape == (batch_size, seq_len, d_model)

def test_residual_connection():
    d_model = 8
    res_pre = ResidualConnection(d_model, dropout=0.0, norm_type="pre")
    res_post = ResidualConnection(d_model, dropout=0.0, norm_type="post")
    
    x = torch.randn(2, 4, d_model)
    sublayer = lambda h: h * 2.0  # Simple linear transformation
    
    out_pre = res_pre(x, sublayer)
    out_post = res_post(x, sublayer)
    
    assert out_pre.shape == x.shape
    assert out_post.shape == x.shape

def test_transformer_encoder_block():
    d_model = 16
    n_heads = 2
    d_ff = 32
    
    block = TransformerEncoderBlock(d_model, n_heads, d_ff, dropout=0.0, norm_type="pre")
    
    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, d_model)
    
    out, weights = block(x)
    
    assert out.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len)

def test_full_transformer_encoder():
    vocab_size = 20
    d_model = 16
    n_heads = 2
    d_ff = 32
    n_layers = 2
    max_seq_len = 10
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        embed_type="learned",
        norm_type="pre",
        dropout=0.1
    )
    
    batch_size = 3
    seq_len = 8
    # Generate random integer token indices
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    out, attn_maps = encoder(input_ids)
    
    assert out.shape == (batch_size, seq_len, d_model)
    assert len(attn_maps) == n_layers
    assert attn_maps[0].shape == (batch_size, n_heads, seq_len, seq_len)
