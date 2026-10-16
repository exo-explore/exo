import importlib

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx import utils_mlx
from exo.worker.engines.mlx.patches.unwired_generation import patch_unwired_generation

_generate_module = importlib.import_module("mlx_lm.generate")


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_stream_generate_leaves_memory_unwired_when_exo_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_unwired_generation()
    model = nn.Linear(4, 4)
    previous = mx.set_wired_limit(0)
    try:
        monkeypatch.setattr(utils_mlx, "_wire_memory", False)
        with _generate_module.wired_limit(model, [_generate_module.generation_stream]):  # pyright: ignore[reportAny]
            assert mx.set_wired_limit(0) == 0

        monkeypatch.setattr(utils_mlx, "_wire_memory", True)
        with _generate_module.wired_limit(model, [_generate_module.generation_stream]):  # pyright: ignore[reportAny]
            assert mx.set_wired_limit(0) > 0
    finally:
        mx.set_wired_limit(previous)
