# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import pytest


def test_constants_exist():
    """Default constants should be defined."""
    from exo.worker.engines.tinygrad.constants import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
    )

    assert isinstance(DEFAULT_MAX_TOKENS, int)
    assert DEFAULT_MAX_TOKENS > 0
    assert isinstance(DEFAULT_TEMPERATURE, float)
    assert isinstance(DEFAULT_TOP_P, float)

def test_initialize_tinygrad_callable():
    """initialize_tinygrad must be importable and callable."""
    from exo.worker.engines.tinygrad.utils_tinygrad import initialize_tinygrad

    assert callable(initialize_tinygrad)

def test_load_tinygrad_items_callable():
    """load_tinygrad_items must be importable and callable."""
    from exo.worker.engines.tinygrad.utils_tinygrad import load_tinygrad_items

    assert callable(load_tinygrad_items)

def test_tinygrad_generate_accepts_prompt_parameter():
    """tinygrad_generate must accept prompt as a parameter (matching MLX pattern)."""
    import inspect

    from exo.worker.engines.tinygrad.generator.generate import tinygrad_generate

    sig = inspect.signature(tinygrad_generate)
    assert "prompt" in sig.parameters, "tinygrad_generate must accept a 'prompt' parameter"
    assert "model" in sig.parameters
    assert "tokenizer" in sig.parameters
    assert "task" in sig.parameters

@pytest.mark.slow
def test_generate_yields_generation_responses():
    """tinygrad_generate should yield GenerationResponse objects."""
    from pathlib import Path
    from unittest.mock import MagicMock

    from exo.shared.model_config import parse_model_config
    from exo.shared.types.worker.runner_response import GenerationResponse
    from exo.worker.engines.tinygrad.generator.generate import tinygrad_generate
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    model_path = Path.home() / ".cache/exo/downloads/mlx-community/Llama-3.2-1B-Instruct-4bit"
    if not model_path.exists():
        pytest.skip("Model not downloaded")

    config = parse_model_config(model_path / "config.json")
    weights = load_transformer_weights(model_path, config, start_layer=0, end_layer=2)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    task = MagicMock()
    task.max_tokens = 3
    task.temperature = 0.0
    task.top_p = 0.9
    task.logprobs = False
    task.top_logprobs = 0

    # Runner computes prompt via apply_chat_template, then passes it
    prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    responses = list(tinygrad_generate(weights, tokenizer, task, prompt=prompt))
    assert len(responses) > 0
    assert all(isinstance(r, GenerationResponse) for r in responses)

    # Last response should have finish_reason, stats, and usage
    assert responses[-1].finish_reason is not None
    assert responses[-1].stats is not None
    assert responses[-1].usage is not None
    assert responses[-1].usage.prompt_tokens > 0
    assert responses[-1].usage.completion_tokens > 0

    # Intermediate responses should have no stats/usage
    if len(responses) > 1:
        assert responses[0].stats is None
        assert responses[0].usage is None

@pytest.mark.slow
def test_generate_populates_logprobs_when_requested():
    """When task.logprobs=True, responses should include logprob and top_logprobs."""
    from pathlib import Path
    from unittest.mock import MagicMock

    from exo.shared.model_config import parse_model_config
    from exo.worker.engines.tinygrad.generator.generate import tinygrad_generate
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    model_path = Path.home() / ".cache/exo/downloads/mlx-community/Llama-3.2-1B-Instruct-4bit"
    if not model_path.exists():
        pytest.skip("Model not downloaded")

    config = parse_model_config(model_path / "config.json")
    weights = load_transformer_weights(model_path, config, start_layer=0, end_layer=2)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    task = MagicMock()
    task.max_tokens = 2
    task.temperature = 0.0
    task.top_p = 0.9
    task.logprobs = True
    task.top_logprobs = 3

    prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    responses = list(tinygrad_generate(weights, tokenizer, task, prompt=prompt))

    for r in responses:
        assert r.logprob is not None, "logprob must be set when task.logprobs=True"
        assert r.logprob <= 0.0, "logprob must be <= 0"
        assert r.top_logprobs is not None
        assert len(r.top_logprobs) == 3

@pytest.mark.slow
def test_generate_omits_logprobs_when_not_requested():
    """When task.logprobs=False, responses should have logprob=None."""
    from pathlib import Path
    from unittest.mock import MagicMock

    from exo.shared.model_config import parse_model_config
    from exo.worker.engines.tinygrad.generator.generate import tinygrad_generate
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    model_path = Path.home() / ".cache/exo/downloads/mlx-community/Llama-3.2-1B-Instruct-4bit"
    if not model_path.exists():
        pytest.skip("Model not downloaded")

    config = parse_model_config(model_path / "config.json")
    weights = load_transformer_weights(model_path, config, start_layer=0, end_layer=2)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    task = MagicMock()
    task.max_tokens = 2
    task.temperature = 0.0
    task.top_p = 0.9
    task.logprobs = False
    task.top_logprobs = 0

    prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    responses = list(tinygrad_generate(weights, tokenizer, task, prompt=prompt))

    for r in responses:
        assert r.logprob is None
        assert r.top_logprobs is None

@pytest.mark.slow
def test_warmup_runs_full_generation():
    """warmup_inference should run a real generation loop, not just a forward pass."""
    from pathlib import Path

    from exo.shared.model_config import parse_model_config
    from exo.worker.engines.tinygrad.generator.generate import warmup_inference
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    model_path = Path.home() / ".cache/exo/downloads/mlx-community/Llama-3.2-1B-Instruct-4bit"
    if not model_path.exists():
        pytest.skip("Model not downloaded")

    config = parse_model_config(model_path / "config.json")
    weights = load_transformer_weights(model_path, config, start_layer=0, end_layer=2)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    tokens_generated = warmup_inference(model=weights, tokenizer=tokenizer)
    assert tokens_generated >= 1, "warmup must generate at least 1 token"
    assert tokens_generated <= 10, "warmup should be short (not full generation)"
