from pathlib import Path

import pytest


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
