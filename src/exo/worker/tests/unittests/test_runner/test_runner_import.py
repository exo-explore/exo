import inspect

import pytest

def _mlx_backend_available() -> bool:
    """Return True only if mlx.core can be fully loaded (native libs present)."""
    try:
        import mlx.core  # noqa: F401
        return True
    except (ImportError, OSError):
        return False

requires_mlx = pytest.mark.skipif(
    not _mlx_backend_available(),
    reason="MLX native backend not available (missing CUDA/Metal libraries)",
)

def test_runner_imports_without_mlx():
    """runner.py must be importable on Linux where MLX is absent."""
    from exo.worker.runner.runner import main

def test_engine_factory_importable():
    """engine_factory.py must be importable on any platform."""
    from exo.worker.engines.engine_factory import Engine, create_engine

def test_engine_is_immutable():
    """Engine must be an immutable dataclass with the expected fields."""
    from exo.worker.engines.engine_factory import Engine
    import dataclasses

    assert dataclasses.is_dataclass(Engine)
    field_names = {f.name for f in dataclasses.fields(Engine)}

    required_fields = [
        "initialize", "load", "generate", "warmup", "cleanup",
        "apply_chat_template", "detect_thinking_prompt_suffix",
    ]

    assert all(field in field_names for field in required_fields)

def test_tokenizer_protocol_importable():
    from exo.shared.types.worker.tokenizer import Tokenizer

@requires_mlx
def test_mlx_engine_has_postprocessing_importable():
    from exo.worker.engines.mlx.generator.generate import mlx_generate_with_postprocessing

@requires_mlx
def test_mlx_engine_has_postprocessing_signature():
    from exo.worker.engines.mlx.generator.generate import mlx_generate_with_postprocessing
    sig = inspect.signature(mlx_generate_with_postprocessing)
    params = list(sig.parameters.keys())

    expected_params = ["model", "tokenizer", "model_id"]

    assert all(param in params for param in expected_params)
