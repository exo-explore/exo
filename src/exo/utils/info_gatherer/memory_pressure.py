from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from typing import NamedTuple

from loguru import logger

from exo.shared.types.profiling import MemoryPressureLevel


class MemoryPressureInfo(NamedTuple):
    """Memory pressure telemetry details."""

    level: MemoryPressureLevel
    pressure_pct: float
    effective_available_bytes: int | None
    wired_bytes: int | None


def _get_darwin_sysctl_pressure() -> tuple[MemoryPressureLevel, float] | None:
    """Read kern.memorystatus_vm_pressure_level on macOS via libc sysctlbyname or CLI."""
    if sys.platform != "darwin":
        return None

    # Try fast C-level sysctlbyname first (sub-microsecond, no subprocess)
    try:
        libc = ctypes.CDLL(None)
        level_val = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(level_val))
        res = libc.sysctlbyname(
            b"kern.memorystatus_vm_pressure_level",
            ctypes.byref(level_val),
            ctypes.byref(size),
            None,
            0,
        )
        if res == 0:
            val = level_val.value
            if val == 1:
                return (MemoryPressureLevel.NORMAL, 20.0)
            elif val == 2:
                return (MemoryPressureLevel.WARNING, 65.0)
            elif val >= 4:
                return (MemoryPressureLevel.CRITICAL, 95.0)
    except (OSError, AttributeError, ValueError) as e:
        logger.debug(f"sysctlbyname unavailable: {e}")

    # Fallback to sysctl CLI
    sysctl_bin = shutil.which("sysctl") or "/usr/sbin/sysctl"
    if os.path.exists(sysctl_bin):
        try:
            out = subprocess.check_output(
                [sysctl_bin, "-n", "kern.memorystatus_vm_pressure_level"],
                timeout=1.5,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            val = int(out)
            if val == 1:
                return (MemoryPressureLevel.NORMAL, 20.0)
            elif val == 2:
                return (MemoryPressureLevel.WARNING, 65.0)
            elif val >= 4:
                return (MemoryPressureLevel.CRITICAL, 95.0)
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            logger.debug(f"Failed to query sysctl memory pressure: {e}")

    return None


def _get_darwin_vm_stats() -> tuple[int | None, int | None]:
    """Parse vm_stat on macOS to retrieve wired bytes and reclaimable purgeable/inactive bytes."""
    if sys.platform != "darwin":
        return None, None

    vm_stat_bin = shutil.which("vm_stat") or "/usr/bin/vm_stat"
    if not os.path.exists(vm_stat_bin):
        return None, None

    try:
        out = subprocess.check_output(
            [vm_stat_bin], timeout=1.5, text=True, stderr=subprocess.DEVNULL
        )
        page_size = 16384 if "page size of 16384 bytes" in out else 4096

        wired_pages = 0
        purgeable_pages = 0
        inactive_pages = 0

        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Pages wired down:"):
                wired_pages = int(line.split(":")[1].replace(".", "").strip())
            elif line.startswith("Pages purgeable:"):
                purgeable_pages = int(line.split(":")[1].replace(".", "").strip())
            elif line.startswith("Pages inactive:"):
                inactive_pages = int(line.split(":")[1].replace(".", "").strip())

        wired_bytes = wired_pages * page_size
        reclaimable_bytes = (purgeable_pages + inactive_pages) * page_size
        return wired_bytes, reclaimable_bytes
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        logger.debug(f"Failed to parse vm_stat: {e}")
        return None, None


def _get_linux_psi_pressure() -> tuple[MemoryPressureLevel, float] | None:
    """Read Linux Pressure Stall Information (PSI) from /proc/pressure/memory."""
    psi_path = "/proc/pressure/memory"
    if not os.path.exists(psi_path):
        return None

    try:
        with open(psi_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("some"):
                    # Format: some avg10=0.00 avg60=0.00 avg300=0.00 total=1234
                    parts = line.split()
                    for p in parts:
                        if p.startswith("avg10="):
                            avg10 = float(p.split("=")[1])
                            if avg10 >= 30.0:
                                return (MemoryPressureLevel.CRITICAL, min(100.0, 75.0 + avg10))
                            elif avg10 >= 8.0:
                                return (MemoryPressureLevel.WARNING, 40.0 + avg10)
                            else:
                                return (MemoryPressureLevel.NORMAL, max(10.0, avg10 * 3.0))
    except (OSError, ValueError) as e:
        logger.debug(f"Failed to read Linux PSI: {e}")

    return None


def probe_memory_pressure(
    ram_total: int,
    psutil_available: int,
) -> MemoryPressureInfo:
    """Probe system memory pressure and compute realistic available RAM.

    Implements official TODO.md #13:
    - Resolves false OOM on macOS where unified memory caches files aggressively.
    - Uses kernel memory pressure level (Normal, Warning, Critical) to guide allocation.
    """
    # 1. Darwin / macOS Memory Pressure
    if sys.platform == "darwin":
        darwin_pressure = _get_darwin_sysctl_pressure()
        wired_bytes, _reclaimable_bytes = _get_darwin_vm_stats()

        level = darwin_pressure[0] if darwin_pressure else MemoryPressureLevel.NORMAL
        pressure_pct = darwin_pressure[1] if darwin_pressure else 25.0

        # Compute effective available memory under memory pressure:
        # Wired memory cannot be paged or purged. The remaining memory (including
        # file caches and inactive pages) can be immediately allocated for Metal / MLX.
        # We preserve a 12% safety margin for OS kernel and foreground window operations.
        if wired_bytes is not None and wired_bytes > 0:
            os_safety_margin = int(ram_total * 0.12)
            usable_headroom = max(0, ram_total - wired_bytes - os_safety_margin)
        else:
            usable_headroom = psutil_available

        if level == MemoryPressureLevel.CRITICAL:
            # Under critical pressure, macOS is heavily swapping/killing. Throttle to 0.
            effective_avail = 0
            pressure_pct = max(pressure_pct, 95.0)
        elif level == MemoryPressureLevel.WARNING:
            # Under warning pressure, only allow conservative allocation.
            effective_avail = min(usable_headroom // 2, psutil_available)
            pressure_pct = max(pressure_pct, 65.0)
        else:
            # Under normal pressure, allow unified memory allocation up to usable headroom.
            # Never return less than psutil_available.
            effective_avail = max(usable_headroom, psutil_available)
            # Cap at 85% of ram_total (Apple Silicon unified memory Metal allocation threshold)
            max_metal_alloc = int(ram_total * 0.85)
            effective_avail = min(effective_avail, max_metal_alloc)

        return MemoryPressureInfo(
            level=level,
            pressure_pct=round(pressure_pct, 1),
            effective_available_bytes=effective_avail,
            wired_bytes=wired_bytes,
        )

    # 2. Linux Pressure Stall Information (PSI)
    if sys.platform.startswith("linux"):
        linux_pressure = _get_linux_psi_pressure()
        if linux_pressure:
            level, pct = linux_pressure
            if level == MemoryPressureLevel.CRITICAL:
                effective_avail = 0
            elif level == MemoryPressureLevel.WARNING:
                effective_avail = psutil_available // 2
            else:
                effective_avail = psutil_available
            return MemoryPressureInfo(
                level=level,
                pressure_pct=round(pct, 1),
                effective_available_bytes=effective_avail,
                wired_bytes=None,
            )

    # 3. Windows / Fallback based on psutil available ratio
    used_ratio = 1.0 - (psutil_available / ram_total) if ram_total > 0 else 0.0
    pressure_pct = round(used_ratio * 100.0, 1)

    if used_ratio >= 0.92:
        level = MemoryPressureLevel.CRITICAL
        effective_avail = 0
    elif used_ratio >= 0.80:
        level = MemoryPressureLevel.WARNING
        effective_avail = psutil_available // 2
    else:
        level = MemoryPressureLevel.NORMAL
        effective_avail = psutil_available

    return MemoryPressureInfo(
        level=level,
        pressure_pct=pressure_pct,
        effective_available_bytes=effective_avail,
        wired_bytes=None,
    )
