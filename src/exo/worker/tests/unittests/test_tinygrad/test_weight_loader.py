from __future__ import annotations

from pathlib import Path

# pyright: reportUnknownMemberType=false
import pytest
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from exo.shared.model_config import ModelConfig, QuantizationConfig

Device.DEFAULT = "CPU"

# ── Helpers ──

def _make_model_config(
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int | None = None,
    intermediate_size: int = 128,
    vocab_size: int = 256,
    quantized: bool = False,
) -> ModelConfig:
    """Build a ModelConfig with small defaults for weight-loader tests."""
    from exo.shared.architecture.llama import LLAMA_SPEC

    n_kv = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
    return ModelConfig(
        architecture_spec=LLAMA_SPEC,
        num_hidden_layers=2,
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
        quantization_config=QuantizationConfig(bits=4, group_size=32) if quantized else None,
    )


def _mlx_quantized_raw(
    key_prefix: str,
    out_features: int,
    in_features: int,
    bits: int = 4,
    group_size: int = 32,
) -> dict[str, Tensor]:
    """Create a synthetic MLX-format quantized weight dict (.weight + .scales + .biases)."""
    pack_factor = 32 // bits
    packed_dim = in_features // pack_factor
    num_groups = in_features // group_size

    return {
        f"{key_prefix}.weight": Tensor.zeros(out_features, packed_dim, dtype=dtypes.uint32),
        f"{key_prefix}.scales": Tensor.ones(out_features, num_groups, dtype=dtypes.float16),
        f"{key_prefix}.biases": Tensor.zeros(out_features, num_groups, dtype=dtypes.float16),
    }


# ── NamedTuple structure tests (pre-existing) ──

def test_layer_weights_is_named_tuple():
    """LayerWeights should be a NamedTuple with expected fields."""
    from exo.worker.engines.tinygrad.weight_loader import LayerWeights

    assert hasattr(LayerWeights, "_fields")
    fields = LayerWeights._fields
    assert "q_proj" in fields
    assert "k_proj" in fields
    assert "v_proj" in fields
    assert "o_proj" in fields
    assert "gate_proj" in fields
    assert "up_proj" in fields
    assert "down_proj" in fields
    assert "input_norm" in fields
    assert "post_attn_norm" in fields

def test_transformer_weights_is_named_tuple():
    """TransformerWeights should contain embed, lm_head, final_norm, layers, config."""
    from exo.worker.engines.tinygrad.weight_loader import TransformerWeights

    fields = TransformerWeights._fields
    assert "embed_tokens" in fields
    assert "lm_head" in fields
    assert "final_norm" in fields
    assert "layers" in fields
    assert "config" in fields

@pytest.mark.slow
def test_load_llama_weights():
    """Load 2 layers from a real Llama model and verify structure."""
    from exo.shared.model_config import parse_model_config
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    model_path = Path.home() / ".cache/exo/downloads/mlx-community/Llama-3.2-1B-Instruct-4bit"
    if not model_path.exists():
        pytest.skip("Model not downloaded")
    config = parse_model_config(model_path / "config.json")
    weights = load_transformer_weights(model_path, config, start_layer=0, end_layer=2)
    assert len(weights.layers) == 2
    assert weights.embed_tokens is not None
    assert weights.final_norm is not None
    assert weights.lm_head is not None

@pytest.mark.slow
def test_load_respects_layer_range():
    """Loading layers 2-4 should produce exactly 2 LayerWeights."""
    from exo.shared.model_config import parse_model_config
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    model_path = Path.home() / ".cache/exo/downloads/mlx-community/Llama-3.2-1B-Instruct-4bit"
    if not model_path.exists():
        pytest.skip("Model not downloaded")
    config = parse_model_config(model_path / "config.json")
    weights = load_transformer_weights(model_path, config, start_layer=2, end_layer=4)
    assert len(weights.layers) == 2

def test_load_missing_safetensors_raises(tmp_path: Path):
    """Loading from an empty directory should raise FileNotFoundError."""
    from exo.worker.engines.tinygrad.weight_loader import (
        _load_all_safetensors,  # pyright: ignore[reportPrivateUsage]
    )

    with pytest.raises(FileNotFoundError, match="No .safetensors"):
        _load_all_safetensors(tmp_path)

# ── _build_weight: MLX quantized format ──

def test_build_weight_mlx_quantized_linear():
    """MLX quantized linear (.weight + .scales + .biases with quantization_config) → QuantizedLinear."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear
    from exo.worker.engines.tinygrad.weight_loader import (
        _build_weight,  # pyright: ignore[reportPrivateUsage]
    )

    config = _make_model_config(quantized=True)
    # o_proj shape: (hidden_size, hidden_size) = (64, 64)
    raw = _mlx_quantized_raw(
        "model.layers.0.self_attn.o_proj", out_features=64, in_features=64,
    )
    result = _build_weight(raw, "model.layers.0.self_attn.o_proj.weight", config)
    assert isinstance(result, QuantizedLinear)

def test_build_weight_mlx_quantized_embedding():
    """MLX quantized embedding (.weight + .scales + .biases with is_embedding=True) → QuantizedEmbedding."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedEmbedding
    from exo.worker.engines.tinygrad.weight_loader import (
        _build_weight,  # pyright: ignore[reportPrivateUsage]
    )

    config = _make_model_config(quantized=True)
    # embed_tokens shape: (vocab_size, hidden_size) = (256, 64)
    raw = _mlx_quantized_raw(
        "model.embed_tokens", out_features=256, in_features=64,
    )
    result = _build_weight(raw, "model.embed_tokens.weight", config, is_embedding=True)
    assert isinstance(result, QuantizedEmbedding)

def test_build_weight_unquantized_returns_tensor():
    """Unquantized model (.weight only, no quantization_config) → plain Tensor."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear
    from exo.worker.engines.tinygrad.weight_loader import (
        _build_weight,  # pyright: ignore[reportPrivateUsage]
    )

    config = _make_model_config(quantized=False)
    raw = {"model.layers.0.self_attn.o_proj.weight": Tensor.zeros(64, 64)}
    result = _build_weight(raw, "model.layers.0.self_attn.o_proj.weight", config)
    assert isinstance(result, Tensor)
    assert not isinstance(result, QuantizedLinear)

def test_build_weight_quantized_config_but_no_companions():
    """Quantized config present but no .scales/.biases companions → plain Tensor (not quantized)."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear
    from exo.worker.engines.tinygrad.weight_loader import (
        _build_weight,  # pyright: ignore[reportPrivateUsage]
    )

    config = _make_model_config(quantized=True)
    # Only .weight, no .scales or .biases
    raw = {"model.layers.0.self_attn.o_proj.weight": Tensor.zeros(64, 64)}
    result = _build_weight(raw, "model.layers.0.self_attn.o_proj.weight", config)
    assert isinstance(result, Tensor)
    assert not isinstance(result, QuantizedLinear)

def test_build_weight_missing_key_raises():
    """Empty dict → KeyError."""
    from exo.worker.engines.tinygrad.weight_loader import (
        _build_weight,  # pyright: ignore[reportPrivateUsage]
    )

    config = _make_model_config(quantized=False)
    with pytest.raises(KeyError):
        _build_weight({}, "model.layers.0.self_attn.o_proj.weight", config)

def test_build_weight_mlx_quantized_shape_correctness():
    """MLX quantized gate_proj should have correct in_features and out_features."""
    from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear
    from exo.worker.engines.tinygrad.weight_loader import (
        _build_weight,  # pyright: ignore[reportPrivateUsage]
    )

    config = _make_model_config(quantized=True)
    # gate_proj shape: (intermediate_size, hidden_size) = (128, 64)
    raw = _mlx_quantized_raw(
        "model.layers.0.mlp.gate_proj", out_features=128, in_features=64,
    )
    result = _build_weight(raw, "model.layers.0.mlp.gate_proj.weight", config)
    assert isinstance(result, QuantizedLinear)
    assert result.out_features == 128
    assert result.in_features == 64
