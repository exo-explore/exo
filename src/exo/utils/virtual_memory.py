"""Memory statistics that survive psutil's Darwin 27 incompatibility.

On macOS 26 (Darwin 27) the kernel grew the ``vm_statistics64`` struct.
psutil (<= 7.2.2) calls ``host_statistics64`` with a buffer sized for the
older struct, and the kernel rejects it with ``(ipc/mig) array not large
enough`` on most calls. Both ``psutil.virtual_memory`` and
``psutil.swap_memory`` are affected.

The helpers here try psutil first and, on Darwin only, fall back to asking
the kernel directly: ``host_statistics64`` with a generously sized buffer
(the struct layout is append-only, so the leading fields keep their
offsets), and ``sysctlbyname`` for totals and swap usage.
"""

import ctypes
import sys
from dataclasses import dataclass
from functools import cache
from typing import cast, final

import psutil

_HOST_VM_INFO64 = 4

# Darwin 27 currently returns 104 naturals; leave plenty of headroom for
# future struct growth since the call fails when the buffer is too small.
_VM_STATISTICS_BUFFER_NATURALS = 1024

# Indices into the vm_statistics64 struct viewed as an array of 32-bit
# naturals. The struct is pragma pack(4): four leading naturals, then nine
# 64-bit counters (zero_fill_count .. purges), then purgeable_count and
# speculative_count. 4 + 9 * 2 = 22.
_FREE_PAGES_INDEX = 0
_INACTIVE_PAGES_INDEX = 2
_SPECULATIVE_PAGES_INDEX = 23
_MINIMUM_NATURALS = _SPECULATIVE_PAGES_INDEX + 1


@final
@dataclass(frozen=True)
class VirtualMemoryStatistics:
    total_bytes: int
    available_bytes: int

    @property
    def used_fraction(self) -> float:
        """Fraction of memory in use, 0.0 - 1.0 (psutil ``percent`` / 100)."""
        if self.total_bytes == 0:
            return 0.0
        return (self.total_bytes - self.available_bytes) / self.total_bytes


@final
@dataclass(frozen=True)
class SwapMemoryStatistics:
    total_bytes: int
    free_bytes: int


def virtual_memory_statistics() -> VirtualMemoryStatistics:
    try:
        virtual_memory = psutil.virtual_memory()
        return VirtualMemoryStatistics(
            total_bytes=virtual_memory.total,
            available_bytes=virtual_memory.available,
        )
    except RuntimeError:
        if sys.platform != "darwin":
            raise
        return _darwin_virtual_memory_statistics()


def swap_memory_statistics() -> SwapMemoryStatistics:
    try:
        swap_memory = psutil.swap_memory()
        return SwapMemoryStatistics(
            total_bytes=swap_memory.total,
            free_bytes=swap_memory.free,
        )
    except RuntimeError:
        if sys.platform != "darwin":
            raise
        return _darwin_swap_memory_statistics()


@final
class _SwapUsage(ctypes.Structure):
    """struct xsw_usage from <sys/sysctl.h>."""

    _fields_ = (
        ("total", ctypes.c_uint64),
        ("avail", ctypes.c_uint64),
        ("used", ctypes.c_uint64),
        ("pagesize", ctypes.c_uint32),
        ("encrypted", ctypes.c_uint32),
    )

    total: int
    avail: int
    used: int
    pagesize: int
    encrypted: int


@cache
def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None)


@cache
def _mach_host_port() -> int:
    libc = _libc()
    libc.mach_host_self.restype = ctypes.c_uint
    return cast(int, libc.mach_host_self())


def _sysctl_by_name(name: str, buffer: ctypes.c_uint64 | _SwapUsage) -> None:
    libc = _libc()
    size = ctypes.c_size_t(ctypes.sizeof(buffer))
    result = cast(
        int,
        libc.sysctlbyname(
            name.encode(), ctypes.byref(buffer), ctypes.byref(size), None, 0
        ),
    )
    if result != 0:
        raise OSError(f"sysctlbyname({name!r}) failed")


@cache
def _darwin_total_memory_bytes() -> int:
    total = ctypes.c_uint64(0)
    _sysctl_by_name("hw.memsize", total)
    return total.value


@cache
def _darwin_page_size_bytes() -> int:
    page_size = ctypes.c_uint64(0)
    _sysctl_by_name("vm.pagesize", page_size)
    return page_size.value


def _darwin_virtual_memory_statistics() -> VirtualMemoryStatistics:
    libc = _libc()
    buffer = (ctypes.c_uint * _VM_STATISTICS_BUFFER_NATURALS)()
    count = ctypes.c_uint(_VM_STATISTICS_BUFFER_NATURALS)
    kern_return = cast(
        int,
        libc.host_statistics64(
            ctypes.c_uint(_mach_host_port()),
            ctypes.c_int(_HOST_VM_INFO64),
            buffer,
            ctypes.byref(count),
        ),
    )
    if kern_return != 0:
        raise OSError(f"host_statistics64(HOST_VM_INFO64) failed: {kern_return}")
    if count.value < _MINIMUM_NATURALS:
        raise OSError(
            f"host_statistics64 returned {count.value} naturals, "
            f"expected at least {_MINIMUM_NATURALS}"
        )

    free_pages = cast(int, buffer[_FREE_PAGES_INDEX])
    inactive_pages = cast(int, buffer[_INACTIVE_PAGES_INDEX])
    speculative_pages = cast(int, buffer[_SPECULATIVE_PAGES_INDEX])

    # Match psutil's definition: available = inactive + (free - speculative).
    available_pages = max(inactive_pages + free_pages - speculative_pages, 0)
    return VirtualMemoryStatistics(
        total_bytes=_darwin_total_memory_bytes(),
        available_bytes=available_pages * _darwin_page_size_bytes(),
    )


def _darwin_swap_memory_statistics() -> SwapMemoryStatistics:
    swap_usage = _SwapUsage()
    _sysctl_by_name("vm.swapusage", swap_usage)
    return SwapMemoryStatistics(
        total_bytes=swap_usage.total,
        free_bytes=swap_usage.avail,
    )
