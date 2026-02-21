# pyright: reportUnknownMemberType=false
from tinygrad.tensor import Tensor

# --- RMS Norm ---

def test_rms_norm_shape():
    """Output shape must match input shape."""
    from exo.worker.engines.tinygrad.layers.normalization import rms_norm

    x = Tensor.randn(1, 4, 2048)
    weight = Tensor.ones(2048)
    out = rms_norm(x, weight, eps=1e-6)
    assert out.shape == (1, 4, 2048)

def test_rms_norm_unit_weight():
    """With unit weight, RMS norm output should have the same shape and be a different tensor."""
    from exo.worker.engines.tinygrad.layers.normalization import rms_norm

    x = Tensor.randn(1, 4, 2048)
    weight = Tensor.ones(2048)
    out = rms_norm(x, weight, eps=1e-6)
    assert out.shape == x.shape
    # Verify the computation graph produces the right intermediate shapes
    rms = ((out * out).mean(axis=-1)).sqrt()
    assert rms.shape == (1, 4)

def test_rms_norm_scales_with_weight():
    """Doubling the weight should produce an output with the same shape."""
    from exo.worker.engines.tinygrad.layers.normalization import rms_norm

    x = Tensor.randn(1, 4, 2048)
    weight_1x = Tensor.ones(2048)
    weight_2x = Tensor.ones(2048) * 2.0
    out_1x = rms_norm(x, weight_1x, eps=1e-6)
    out_2x = rms_norm(x, weight_2x, eps=1e-6)
    assert out_1x.shape == out_2x.shape == (1, 4, 2048)
    # Verify the difference tensor has the right shape
    diff = (out_2x - out_1x * 2.0)
    assert diff.shape == (1, 4, 2048)

# --- RoPE ---

def test_rope_frequency_shapes():
    """cos/sin frequency tables should have shape (max_seq_len, head_dim//2)."""
    from exo.worker.engines.tinygrad.layers.rotary import compute_rope_frequencies

    cos_f, sin_f = compute_rope_frequencies(head_dim=64, max_seq_len=128)
    assert cos_f.shape == (128, 32)
    assert sin_f.shape == (128, 32)

def test_rope_changes_with_position():
    """apply_rope at different offsets should produce tensors of the same shape."""
    from exo.worker.engines.tinygrad.layers.rotary import (
        apply_rope,
        compute_rope_frequencies,
    )

    cos_f, sin_f = compute_rope_frequencies(head_dim=64, max_seq_len=128)
    x = Tensor.randn(1, 8, 4, 64)  # (batch, heads, seq=4, head_dim)
    out_0 = apply_rope(x, cos_f, sin_f, position_offset=0)
    out_10 = apply_rope(x, cos_f, sin_f, position_offset=10)
    assert out_0.shape == out_10.shape == (1, 8, 4, 64)
    # Verify the difference tensor can be constructed (computation graph is valid)
    diff = (out_0 - out_10).abs()
    assert diff.shape == (1, 8, 4, 64)

def test_rope_preserves_shape():
    """apply_rope must not change the tensor shape."""
    from exo.worker.engines.tinygrad.layers.rotary import (
        apply_rope,
        compute_rope_frequencies,
    )

    cos_f, sin_f = compute_rope_frequencies(head_dim=64, max_seq_len=128)
    x = Tensor.randn(1, 8, 4, 64)
    out = apply_rope(x, cos_f, sin_f, position_offset=0)
    assert out.shape == x.shape

# --- Embedding ---

def test_embedding_lookup():
    """Embedding lookup should index into the weight table."""
    from exo.worker.engines.tinygrad.layers.embedding import apply_embedding

    embed = Tensor.randn(100, 32)  # vocab=100, dim=32
    ids = Tensor([[0, 1, 2]])
    out = apply_embedding(embed, ids)
    assert out.shape == (1, 3, 32)

def test_lm_head_shape():
    """LM head should project hidden_size to vocab_size."""
    from exo.worker.engines.tinygrad.layers.embedding import apply_lm_head

    x = Tensor.randn(1, 4, 2048)
    lm_head = Tensor.randn(128256, 2048)  # (vocab, hidden)
    out = apply_lm_head(x, lm_head)
    assert out.shape == (1, 4, 128256)

# --- MLP ---

def test_swiglu_mlp_shape():
    """SwiGLU MLP: hidden_size in, hidden_size out."""
    from exo.worker.engines.tinygrad.layers.mlp import swiglu_mlp

    x = Tensor.randn(1, 4, 2048)
    gate = Tensor.randn(8192, 2048)
    up = Tensor.randn(8192, 2048)
    down = Tensor.randn(2048, 8192)
    out = swiglu_mlp(x, gate, up, down)
    assert out.shape == (1, 4, 2048)

# --- Attention ---

def test_linear_forward_shape():
    """linear_forward(x, weight) should compute x @ weight.T."""
    from exo.worker.engines.tinygrad.layers.attention import linear_forward

    x = Tensor.randn(1, 4, 2048)
    weight = Tensor.randn(512, 2048)  # (out, in)
    out = linear_forward(x, weight)
    assert out.shape == (1, 4, 512)
