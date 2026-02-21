# pyright: reportUnknownMemberType=false
import pytest
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


@pytest.mark.slow
def test_forward_pass_shape():
    """Forward pass should produce logits of shape (batch, seq, vocab_size)."""
    from pathlib import Path

    from exo.shared.model_config import parse_model_config
    from exo.worker.engines.tinygrad.forward import forward_pass
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    model_path = Path.home() / ".cache/exo/downloads/mlx-community/Llama-3.2-1B-Instruct-4bit"
    if not model_path.exists():
        pytest.skip("Model not downloaded")
    config = parse_model_config(model_path / "config.json")
    weights = load_transformer_weights(model_path, config, start_layer=0, end_layer=2)

    input_ids = Tensor([[1, 2, 3, 4]], dtype=dtypes.int32)
    _logits, cache = forward_pass(weights, input_ids, cache=None, position_offset=0)
    assert _logits.shape == (1, 4, config.vocab_size)
    assert cache.seq_len == 4

@pytest.mark.slow
def test_decode_step_after_prefill():
    """A single-token decode step should extend the cache by 1."""
    from pathlib import Path

    from exo.shared.model_config import parse_model_config
    from exo.worker.engines.tinygrad.forward import forward_pass
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    model_path = Path.home() / ".cache/exo/downloads/mlx-community/Llama-3.2-1B-Instruct-4bit"
    if not model_path.exists():
        pytest.skip("Model not downloaded")
    config = parse_model_config(model_path / "config.json")
    weights = load_transformer_weights(model_path, config, start_layer=0, end_layer=2)

    input_ids = Tensor([[1, 2, 3, 4]], dtype=dtypes.int32)
    _logits, cache = forward_pass(weights, input_ids, cache=None, position_offset=0)

    next_input = Tensor([[5]], dtype=dtypes.int32)
    logits2, cache2 = forward_pass(weights, next_input, cache, position_offset=4)
    assert logits2.shape == (1, 1, config.vocab_size)
    assert cache2.seq_len == 5
