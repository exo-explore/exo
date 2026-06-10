import sys

import pytest

from exo.utils.virtual_memory import (
    SwapMemoryStatistics,
    VirtualMemoryStatistics,
    swap_memory_statistics,
    virtual_memory_statistics,
)


def test_virtual_memory_statistics_returns_sane_values():
    statistics = virtual_memory_statistics()
    assert statistics.total_bytes > 0
    assert 0 < statistics.available_bytes <= statistics.total_bytes
    assert 0.0 <= statistics.used_fraction <= 1.0


def test_virtual_memory_statistics_never_raises():
    # On Darwin 27 psutil.virtual_memory() fails intermittently with
    # "host_statistics64 ... (ipc/mig) array not large enough"; the
    # fallback must absorb that on every call.
    for _ in range(30):
        virtual_memory_statistics()


def test_swap_memory_statistics_returns_sane_values():
    statistics = swap_memory_statistics()
    assert statistics.total_bytes >= 0
    assert 0 <= statistics.free_bytes <= max(statistics.total_bytes, 1)


@pytest.mark.skipif(sys.platform != "darwin", reason="darwin-only fallback")
def test_darwin_fallback_matches_macos_memory_shape():
    from exo.utils.virtual_memory import (
        _darwin_swap_memory_statistics,  # pyright: ignore[reportPrivateUsage]
        _darwin_virtual_memory_statistics,  # pyright: ignore[reportPrivateUsage]
    )

    virtual_memory = _darwin_virtual_memory_statistics()
    assert isinstance(virtual_memory, VirtualMemoryStatistics)
    assert virtual_memory.total_bytes > 2**30
    assert 0 < virtual_memory.available_bytes <= virtual_memory.total_bytes

    swap_memory = _darwin_swap_memory_statistics()
    assert isinstance(swap_memory, SwapMemoryStatistics)
    assert swap_memory.free_bytes <= swap_memory.total_bytes or (
        swap_memory.total_bytes == 0 and swap_memory.free_bytes == 0
    )


def test_used_fraction_handles_zero_total():
    assert (
        VirtualMemoryStatistics(total_bytes=0, available_bytes=0).used_fraction == 0.0
    )
