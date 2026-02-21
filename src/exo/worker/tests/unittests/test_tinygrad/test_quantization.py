from __future__ import annotations

# pyright: reportUnknownMemberType=false
import pytest
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from exo.shared.model_config import ModelConfig

Device.DEFAULT = "CPU"

# ── Helpers ──

def _make_model_config(
    hidden_size: int = 2048,
    num_attention_heads: int = 32,
    num_key_value_heads: int | None = None,
    intermediate_size: int = 8192,
    vocab_size: int = 128256,
) -> ModelConfig:
    """Build a ModelConfig with sensible defaults for quantization tests."""
    from exo.shared.architecture.llama import LLAMA_SPEC

    n_kv = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
    return ModelConfig(
        architecture_spec=LLAMA_SPEC,
        num_hidden_layers=32,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=n_kv,
        vocab_size=vocab_size,
        head_dim=hidden_size // num_attention_heads,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        quantization_config=None,
    )

# ── Packing / Unpacking ──

def test_calculate_pack_factor():
    """pack_factor = 32 // bits."""
    from exo.worker.engines.tinygrad.quantization.packing import calculate_pack_factor

    assert calculate_pack_factor(4) == 8
    assert calculate_pack_factor(8) == 4
    assert calculate_pack_factor(2) == 16

def test_pack_bits_output_shape():
    """Packed last dim should be original_last_dim / pack_factor."""
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits

    original = Tensor.zeros(64, 4096, dtype=dtypes.float16)
    packed = pack_bits(original, bits=4)
    assert packed.tensor.shape == (64, 512)

def test_packed_tensor_metadata():
    """PackedTensor must carry original_shape, pack_factor, bits."""
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits

    original = Tensor.zeros(64, 4096, dtype=dtypes.float16)
    packed = pack_bits(original, bits=4)
    assert packed.original_shape == (64, 4096)
    assert packed.pack_factor == 8
    assert packed.bits == 4

def test_packed_tensor_is_frozen():
    """PackedTensor must be immutable (frozen dataclass)."""
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits

    packed = pack_bits(Tensor.zeros(4, 8, dtype=dtypes.float16), bits=4)
    with pytest.raises(AttributeError):
        packed.bits = 8

def test_pack_unpack_roundtrip_4bit():
    """pack then unpack should recover original values (4-bit)."""
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits, unpack_bits

    original = Tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=dtypes.float16)
    packed = pack_bits(original, bits=4)
    unpacked = unpack_bits(packed)
    assert unpacked.shape == original.shape
    for i in range(8):
        assert abs(unpacked[0, i].item() - original[0, i].item()) < 1e-3

def test_pack_unpack_roundtrip_8bit():
    """pack then unpack should recover original values (8-bit)."""
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits, unpack_bits

    original = Tensor([[10, 20, 30, 40]], dtype=dtypes.float16)
    packed = pack_bits(original, bits=8)
    unpacked = unpack_bits(packed)
    assert unpacked.shape == original.shape
    for i in range(4):
        assert abs(unpacked[0, i].item() - original[0, i].item()) < 1e-3

def test_pack_unpack_non_divisible_dim():
    """Unpacking handles last dims not divisible by pack_factor."""
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits, unpack_bits

    original = Tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=dtypes.float16)
    packed = pack_bits(original, bits=4)
    unpacked = unpack_bits(packed)
    assert unpacked.shape == (1, 10)
    for i in range(10):
        assert abs(unpacked[0, i].item() - original[0, i].item()) < 1e-3

def test_unpack_dtype_is_float16():
    """Unpacked tensor must be float16."""
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits, unpack_bits

    original = Tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtypes.float16)
    unpacked = unpack_bits(pack_bits(original, bits=4))
    assert unpacked.dtype == dtypes.float16

# ── Dequantization ──

def test_affine_dequantize_identity():
    """scale=1, bias=0 → identity."""
    from exo.worker.engines.tinygrad.quantization.dequantization import (
        affine_dequantize,
    )

    quantized = Tensor([[1.0, 2.0, 3.0, 4.0]])
    scales = Tensor([[1.0]])
    biases = Tensor([[0.0]])
    result = affine_dequantize(quantized, scales, biases, group_size=4)
    assert result.shape == (1, 4)
    for i in range(4):
        assert abs(result[0, i].item() - quantized[0, i].item()) < 1e-5

def test_affine_dequantize_scaling():
    """scale=2 should double values."""
    from exo.worker.engines.tinygrad.quantization.dequantization import (
        affine_dequantize,
    )

    quantized = Tensor([[1.0, 2.0, 3.0, 4.0]])
    scales = Tensor([[2.0]])
    biases = Tensor([[0.0]])
    result = affine_dequantize(quantized, scales, biases, group_size=4)
    for i in range(4):
        assert abs(result[0, i].item() - quantized[0, i].item() * 2.0) < 1e-5

def test_affine_dequantize_bias_offset():
    """bias=10 should offset by 10."""
    from exo.worker.engines.tinygrad.quantization.dequantization import (
        affine_dequantize,
    )

    quantized = Tensor([[0.0, 0.0, 0.0, 0.0]])
    scales = Tensor([[1.0]])
    biases = Tensor([[10.0]])
    result = affine_dequantize(quantized, scales, biases, group_size=4)
    for i in range(4):
        assert abs(result[0, i].item() - 10.0) < 1e-5

def test_affine_dequantize_multiple_groups():
    """Each group uses its own scale and bias."""
    from exo.worker.engines.tinygrad.quantization.dequantization import (
        affine_dequantize,
    )

    quantized = Tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    scales = Tensor([[2.0, 3.0]])   # 2 groups of 4
    biases = Tensor([[0.0, 10.0]])
    result = affine_dequantize(quantized, scales, biases, group_size=4)
    assert abs(result[0, 0].item() - 2.0) < 1e-5    # group 0: 1*2+0
    assert abs(result[0, 4].item() - 13.0) < 1e-5   # group 1: 1*3+10

def test_affine_dequantize_2d_weight_matrix():
    """Dequantize a realistic 2D weight matrix [out_features, in_features]."""
    from exo.worker.engines.tinygrad.quantization.dequantization import (
        affine_dequantize,
    )

    out_features, in_features, group_size = 32, 64, 32
    quantized = Tensor.ones(out_features, in_features)
    num_groups = in_features // group_size
    scales = Tensor.ones(out_features, num_groups) * 0.5
    biases = Tensor.zeros(out_features, num_groups)
    result = affine_dequantize(quantized, scales, biases, group_size=group_size)
    assert result.shape == (out_features, in_features)
    assert abs(result[0, 0].item() - 0.5) < 1e-5

# ── QuantizedLinear ──

def test_quantized_linear_is_callable():
    """QuantizedLinear instances must be callable via __call__."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits

    weight = Tensor.ones(16, 32, dtype=dtypes.float16)
    packed = pack_bits(weight, bits=4)
    layer = QuantizedLinear(
        weight_q=packed,
        scales=Tensor.ones(16, 1),
        biases=Tensor.zeros(16, 1),
        group_size=32,
    )
    x = Tensor.randn(1, 4, 32)
    out = layer(x)
    assert out.shape == (1, 4, 16)

def test_quantized_linear_output_shape():
    """QuantizedLinear(in=64, out=32) → (batch, seq, 32)."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits

    weight = Tensor.ones(32, 64, dtype=dtypes.float16)
    packed = pack_bits(weight, bits=4)
    layer = QuantizedLinear(
        weight_q=packed,
        scales=Tensor.ones(32, 1),
        biases=Tensor.zeros(32, 1),
        group_size=64,
    )
    x = Tensor.randn(1, 8, 64)
    out = layer(x)
    assert out.shape == (1, 8, 32)

def test_quantized_linear_caches_dequantized_weights():
    """Second call reuses cached weights."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits

    weight = Tensor.ones(16, 32, dtype=dtypes.float16)
    packed = pack_bits(weight, bits=4)
    layer = QuantizedLinear(
        weight_q=packed,
        scales=Tensor.ones(16, 1),
        biases=Tensor.zeros(16, 1),
        group_size=32,
    )
    x = Tensor.randn(1, 1, 32)
    layer(x)
    assert layer._dequantized_weight is not None  # pyright: ignore[reportPrivateUsage]
    cached_ref = layer._dequantized_weight  # pyright: ignore[reportPrivateUsage]
    layer(x)
    assert layer._dequantized_weight is cached_ref  # pyright: ignore[reportPrivateUsage]

def test_quantized_linear_clear_cache():
    """clear_cache() resets the dequantized weight cache."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits

    weight = Tensor.ones(16, 32, dtype=dtypes.float16)
    packed = pack_bits(weight, bits=4)
    layer = QuantizedLinear(
        weight_q=packed,
        scales=Tensor.ones(16, 1),
        biases=Tensor.zeros(16, 1),
        group_size=32,
    )
    layer(Tensor.randn(1, 1, 32))
    assert layer._dequantized_weight is not None  # pyright: ignore[reportPrivateUsage]
    layer.clear_cache()
    assert layer._dequantized_weight is None  # pyright: ignore[reportPrivateUsage]

def test_quantized_linear_in_out_features():
    """in_features and out_features reflect the original weight shape."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits

    weight = Tensor.ones(128, 256, dtype=dtypes.float16)
    packed = pack_bits(weight, bits=4)
    layer = QuantizedLinear(
        weight_q=packed,
        scales=Tensor.ones(128, 4),
        biases=Tensor.zeros(128, 4),
        group_size=64,
    )
    assert layer.out_features == 128
    assert layer.in_features == 256

def test_quantized_linear_no_bias_by_default():
    """QuantizedLinear should not add a bias term by default."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits

    weight = Tensor.ones(16, 32, dtype=dtypes.float16)
    packed = pack_bits(weight, bits=4)
    layer = QuantizedLinear(
        weight_q=packed,
        scales=Tensor.ones(16, 1),
        biases=Tensor.zeros(16, 1),
        group_size=32,
    )
    assert layer.bias is None

def test_quantized_linear_is_final():
    """QuantizedLinear must be decorated with @final."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear

    assert getattr(QuantizedLinear, "__final__", False)

# ── QuantizedEmbedding ──

def test_quantized_embedding_lookup():
    """QuantizedEmbedding must return correct shape for index lookup."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedEmbedding
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits

    weight = Tensor.ones(100, 32, dtype=dtypes.float16)
    packed = pack_bits(weight, bits=4)
    layer = QuantizedEmbedding(
        num_embeddings=100,
        embedding_dim=32,
        weight_q=packed,
        scales=Tensor.ones(100, 1),
        biases=Tensor.zeros(100, 1),
        group_size=32,
    )
    ids = Tensor([[0, 1, 2]])
    out = layer(ids)
    assert out.shape == (1, 3, 32)

def test_quantized_embedding_clear_cache():
    """clear_cache() resets the dequantized embedding cache."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedEmbedding
    from exo.worker.engines.tinygrad.quantization.packing import pack_bits

    weight = Tensor.ones(100, 32, dtype=dtypes.float16)
    packed = pack_bits(weight, bits=4)
    layer = QuantizedEmbedding(
        num_embeddings=100,
        embedding_dim=32,
        weight_q=packed,
        scales=Tensor.ones(100, 1),
        biases=Tensor.zeros(100, 1),
        group_size=32,
    )
    layer(Tensor([[0]]))
    assert layer._dequantized_weight is not None  # pyright: ignore[reportPrivateUsage]
    layer.clear_cache()
    assert layer._dequantized_weight is None  # pyright: ignore[reportPrivateUsage]

def test_quantized_embedding_is_final():
    """QuantizedEmbedding must be decorated with @final."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedEmbedding

    assert getattr(QuantizedEmbedding, "__final__", False)

# ── Shape Inference (requires Phase 1 ModelConfig) ──

def test_infer_shape_q_proj():
    """q_proj shape = (hidden_size, hidden_size)."""
    from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape

    config = _make_model_config(hidden_size=2048, num_attention_heads=32)
    shape = infer_weight_shape("model.layers.0.self_attn.q_proj.weight", config)
    assert shape == (2048, 2048)

def test_infer_shape_k_proj_gqa():
    """k_proj with GQA = (num_kv_heads * head_dim, hidden_size)."""
    from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape

    config = _make_model_config(
        hidden_size=2048, num_attention_heads=32, num_key_value_heads=8,
    )
    # head_dim = 2048/32 = 64, kv_dim = 8*64 = 512
    shape = infer_weight_shape("model.layers.0.self_attn.k_proj.weight", config)
    assert shape == (512, 2048)

def test_infer_shape_v_proj_gqa():
    """v_proj with GQA = (num_kv_heads * head_dim, hidden_size)."""
    from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape

    config = _make_model_config(
        hidden_size=2048, num_attention_heads=32, num_key_value_heads=8,
    )
    shape = infer_weight_shape("model.layers.0.self_attn.v_proj.weight", config)
    assert shape == (512, 2048)

def test_infer_shape_k_proj_mha():
    """When num_kv_heads == num_heads (MHA), k_proj = (hidden, hidden)."""
    from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape

    config = _make_model_config(
        hidden_size=2048, num_attention_heads=32, num_key_value_heads=32,
    )
    shape = infer_weight_shape("model.layers.0.self_attn.k_proj.weight", config)
    assert shape == (2048, 2048)

def test_detect_layer_type_v_proj():
    """Must detect v_proj correctly (not w_proj)."""
    from exo.worker.engines.tinygrad.quantization.shapes import detect_layer_type

    assert detect_layer_type("model.layers.0.self_attn.v_proj.weight") == "v_proj"

def test_infer_shape_gate_proj():
    """gate_proj = (intermediate_size, hidden_size)."""
    from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape

    config = _make_model_config(hidden_size=2048, intermediate_size=8192)
    shape = infer_weight_shape("model.layers.0.mlp.gate_proj.weight", config)
    assert shape == (8192, 2048)

def test_infer_shape_down_proj():
    """down_proj = (hidden_size, intermediate_size)."""
    from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape

    config = _make_model_config(hidden_size=2048, intermediate_size=8192)
    shape = infer_weight_shape("model.layers.0.mlp.down_proj.weight", config)
    assert shape == (2048, 8192)

def test_infer_shape_embed_tokens():
    """embed_tokens = (vocab_size, hidden_size)."""
    from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape

    config = _make_model_config(hidden_size=2048, vocab_size=128256)
    shape = infer_weight_shape("model.embed_tokens.weight", config)
    assert shape == (128256, 2048)


def test_infer_shape_lm_head():
    """lm_head = (vocab_size, hidden_size)."""
    from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape

    config = _make_model_config(hidden_size=2048, vocab_size=128256)
    shape = infer_weight_shape("lm_head.weight", config)
    assert shape == (128256, 2048)

# ── Package Exports ──

def test_package_exports_quantized_linear():
    from exo.worker.engines.tinygrad.quantization import (
        QuantizedLinear,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    )

def test_package_exports_quantized_embedding():
    from exo.worker.engines.tinygrad.quantization import (
        QuantizedEmbedding,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    )

def test_package_exports_packed_tensor():
    from exo.worker.engines.tinygrad.quantization import (
        PackedTensor,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    )
