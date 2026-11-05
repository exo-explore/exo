# pyright: reportPrivateUsage=false
"""JACCL instances leave memory unwired; everything else wires the model."""

import mlx.core as mx
import pytest

from exo.shared.types.memory import Memory
from exo.worker.engines.mlx import utils_mlx


@pytest.mark.parametrize("wire_memory", [True, False])
def test_set_wired_limit_for_model_respects_backend(
    monkeypatch: pytest.MonkeyPatch, wire_memory: bool
) -> None:
    calls: list[int] = []

    def record(limit: int) -> int:
        calls.append(limit)
        return 0

    monkeypatch.setattr(mx, "set_wired_limit", record)
    monkeypatch.setattr(utils_mlx, "_wire_memory", wire_memory)

    utils_mlx.set_wired_limit_for_model(Memory.from_bytes(1024))

    if not mx.metal.is_available():
        assert calls == []
    elif wire_memory:
        assert calls == [int(mx.device_info()["max_recommended_working_set_size"])]
    else:
        assert calls == []
