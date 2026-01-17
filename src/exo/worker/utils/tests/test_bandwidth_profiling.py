"""Tests for memory bandwidth profiling."""

import platform
import pytest
from exo.worker.utils.profile import get_memory_bandwidth


@pytest.mark.skipif(
    platform.system().lower() != "darwin",
    reason="MLX bandwidth profiling only works on macOS",
)
async def test_get_memory_bandwidth() -> None:
    """get_memory_bandwidth should return a positive bandwidth value."""
    bandwidth = await get_memory_bandwidth()
    assert bandwidth is not None
    assert bandwidth > 0
