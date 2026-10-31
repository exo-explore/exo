"""MLX's memory limit sits just under the GPU's working set, unless the user sets one."""

import mlx.core as mx
import pytest

from exo.shared.types.memory import Memory
from exo.worker.engines.mlx.utils_mlx import (
    MEMORY_LIMIT_ENV,
    MEMORY_LIMIT_MARGIN,
    memory_limit_for_working_set,
    set_wired_limit_for_model,
)

WORKING_SET = Memory.from_gb(83.5)


def test_default_limit_is_under_the_working_set() -> None:
    limit = memory_limit_for_working_set(WORKING_SET, None)
    assert limit is not None
    assert limit == WORKING_SET - MEMORY_LIMIT_MARGIN
    assert limit < WORKING_SET


@pytest.mark.parametrize(("override", "expected_gb"), [("", None), ("60", 60.0)])
def test_user_limit(override: str, expected_gb: float | None) -> None:
    limit = memory_limit_for_working_set(WORKING_SET, override)
    if expected_gb is None:
        assert limit == WORKING_SET - MEMORY_LIMIT_MARGIN
    else:
        assert limit == Memory.from_gb(expected_gb)


def test_zero_keeps_mlx_default() -> None:
    assert memory_limit_for_working_set(WORKING_SET, "0") is None


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_runner_sets_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MEMORY_LIMIT_ENV, raising=False)
    previous = mx.set_memory_limit(int(mx.device_info()["memory_size"]))
    try:
        set_wired_limit_for_model(Memory.from_gb(1))
        working_set = Memory.from_bytes(
            int(mx.device_info()["max_recommended_working_set_size"])
        )
        expected = (working_set - MEMORY_LIMIT_MARGIN).in_bytes
        assert mx.set_memory_limit(previous) == expected
    finally:
        mx.set_memory_limit(previous)
        mx.set_wired_limit(0)
