"""Tests for memory bandwidth profiling."""

import platform
import pytest
from exo.utils.info_gatherer.info_gatherer import NodeMemoryBandwidth


@pytest.mark.skipif(
    platform.system().lower() != "darwin",
    reason="MLX bandwidth profiling only works on macOS",
)
async def test_get_memory_bandwidth() -> None:
    """NodeMemoryBandwidth.gather should return a positive bandwidth value."""
    bandwidth_info = await NodeMemoryBandwidth.gather()
    assert bandwidth_info.memory_bandwidth is not None
    assert bandwidth_info.memory_bandwidth > 0
