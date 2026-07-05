import pytest
import torch
import math
from logmind.models.attention import ScaledDotProductAttention, MultiHeadSelfAttention

def test_scaled_dot_product_attention_shapes():
    attn = ScaledDotProductAttention(dropout=0.0)
    
    batch_size = 2
    num_heads = 4
    seq_len = 8
    d_k = 16
    
    q = torch.randn(batch_size, num_heads, seq_len, d_k)
    k = torch.randn(batch_size, num_heads, seq_len, d_k)
    v = torch.randn(batch_size, num_heads, seq_len, d_k)
    
    context, weights = attn(q, k, v)
    
    assert context.shape == (batch_size, num_heads, seq_len, d_k)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # Softmax check: weights should sum to 1.0 along the last dimension
    sum_weights = weights.sum(dim=-1)
    assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-6)

def test_scaled_dot_product_attention_mask():
    attn = ScaledDotProductAttention(dropout=0.0)
    
    batch_size = 1
    num_heads = 1
    seq_len = 4
    d_k = 8
    
    q = torch.randn(batch_size, num_heads, seq_len, d_k)
    k = torch.randn(batch_size, num_heads, seq_len, d_k)
    v = torch.randn(batch_size, num_heads, seq_len, d_k)
    
    # Create mask where token index 3 is completely masked out for all positions
    # Shape: (batch_size, 1, seq_len, seq_len)
    mask = torch.zeros(batch_size, 1, seq_len, seq_len)
    mask[:, :, :, 3] = -1e9  # Large negative value to mask out key at index 3
    
    _, weights = attn(q, k, v, mask=mask)
    
    # The attention weights for index 3 should be 0.0
    assert torch.allclose(weights[:, :, :, 3], torch.zeros_like(weights[:, :, :, 3]), atol=1e-6)

def test_multi_head_self_attention_shapes():
    d_model = 32
    n_heads = 4
    mha = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    
    batch_size = 3
    seq_len = 10
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, weights = mha(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len)

def test_mha_gradients():
    d_model = 16
    n_heads = 2
    mha = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    
    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    output, _ = mha(x)
    loss = output.sum()
    loss.backward()
    
    # Verify gradients flow back to inputs and parameters
    assert x.grad is not None
    assert mha.w_q.grad is not None
    assert mha.b_q.grad is not None
    assert mha.w_o.grad is not None
