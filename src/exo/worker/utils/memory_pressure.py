"""Memory pressure detection for macOS and Linux.

This module provides platform-specific memory pressure detection:
- macOS: Uses sysctl vm.memory_pressure and memory_pressure command
- Linux: Uses PSI (Pressure Stall Information) from /proc/pressure/memory

References:
- macOS: https://support.apple.com/guide/activity-monitor/actmntr1004/mac
- Linux PSI: https://docs.kernel.org/accounting/psi.html
- Facebook PSI: https://facebookmicrosites.github.io/psi/docs/overview
"""

from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path

import anyio
from loguru import logger

from exo.shared.types.profiling import MemoryPressureLevel, PSIMetrics


async def get_memory_pressure() -> tuple[MemoryPressureLevel, float, PSIMetrics | None]:
    """Get current memory pressure level for the platform.

    Returns:
        A tuple of (pressure_level, free_pct, psi_metrics):
        - pressure_level: MemoryPressureLevel enum
        - free_pct: System-wide free memory percentage (macOS) or 100 - some_avg10 (Linux)
        - psi_metrics: PSIMetrics on Linux, None on other platforms
    """
    system = platform.system().lower()

    if system == "darwin":
        level, free_pct = await _get_macos_memory_pressure()
        return level, free_pct, None
    elif system == "linux":
        psi = await _get_linux_psi_metrics()
        if psi is not None:
            level = psi.to_pressure_level()
            # Approximate "free" as inverse of pressure
            free_pct = max(0.0, 100.0 - psi.some_avg10)
            return level, free_pct, psi
        # PSI not available (old kernel), fall back to normal
        return MemoryPressureLevel.NORMAL, 100.0, None
    else:
        # Unsupported platform, assume normal
        return MemoryPressureLevel.NORMAL, 100.0, None


async def _get_macos_memory_pressure() -> tuple[MemoryPressureLevel, float]:
    """Get memory pressure on macOS using sysctl and memory_pressure command.

    Uses two sources:
    1. sysctl vm.memory_pressure - direct kernel pressure level (0, 1, 2, or 4)
    2. memory_pressure command - system-wide free percentage

    Returns:
        A tuple of (pressure_level, free_percentage)
    """
    level = MemoryPressureLevel.NORMAL
    free_pct = 100.0

    # Method 1: Try sysctl vm.memory_pressure (fast, direct)
    try:
        result = await anyio.run_process(
            ["sysctl", "-n", "vm.memory_pressure"],
            check=False,
        )
        if result.returncode == 0:
            value = int(result.stdout.decode().strip())
            # macOS kernel values: 0=unknown, 1=normal, 2=warn, 4=critical
            if value == 4:
                level = MemoryPressureLevel.CRITICAL
            elif value == 2:
                level = MemoryPressureLevel.WARN
            elif value >= 1:
                level = MemoryPressureLevel.NORMAL
            # value 0 means we should fall back to memory_pressure command
    except (subprocess.SubprocessError, ValueError) as e:
        logger.debug(f"sysctl vm.memory_pressure failed: {e}")

    # Method 2: Get free percentage from memory_pressure command
    try:
        result = await anyio.run_process(
            ["memory_pressure"],
            check=False,
        )
        if result.returncode == 0:
            output = result.stdout.decode()
            # Parse: "System-wide memory free percentage: 33%"
            match = re.search(r"free percentage:\s*(\d+)%", output)
            if match:
                free_pct = float(match.group(1))

                # If sysctl returned 0 (unknown), derive level from free percentage
                if level == MemoryPressureLevel.NORMAL and free_pct < 20:
                    level = MemoryPressureLevel.WARN
                if free_pct < 10:
                    level = MemoryPressureLevel.CRITICAL
    except (subprocess.SubprocessError, ValueError) as e:
        logger.debug(f"memory_pressure command failed: {e}")

    return level, free_pct


async def _get_linux_psi_metrics() -> PSIMetrics | None:
    """Read Linux PSI (Pressure Stall Information) metrics.

    Parses /proc/pressure/memory which has format:
        some avg10=0.00 avg60=0.00 avg300=0.00 total=0
        full avg10=0.00 avg60=0.00 avg300=0.00 total=0

    Returns:
        PSIMetrics if available, None if PSI is not supported
    """
    psi_path = Path("/proc/pressure/memory")

    if not psi_path.exists():
        logger.debug("PSI not available: /proc/pressure/memory does not exist")
        return None

    try:
        content = await anyio.Path(psi_path).read_text()
        return _parse_psi_content(content)
    except (OSError, PermissionError) as e:
        logger.debug(f"Failed to read PSI metrics: {e}")
        return None


def _parse_psi_content(content: str) -> PSIMetrics:
    """Parse PSI file content into PSIMetrics.

    Args:
        content: Raw content from /proc/pressure/memory

    Returns:
        PSIMetrics with parsed values
    """
    metrics = PSIMetrics()

    for line in content.strip().split("\n"):
        parts = line.split()
        if not parts:
            continue

        prefix = parts[0]  # 'some' or 'full'
        values: dict[str, float] = {}

        for part in parts[1:]:
            if "=" in part:
                key, val = part.split("=", 1)
                try:
                    values[key] = float(val)
                except ValueError:
                    continue

        if prefix == "some":
            metrics.some_avg10 = values.get("avg10", 0.0)
            metrics.some_avg60 = values.get("avg60", 0.0)
            metrics.some_avg300 = values.get("avg300", 0.0)
        elif prefix == "full":
            metrics.full_avg10 = values.get("avg10", 0.0)
            metrics.full_avg60 = values.get("avg60", 0.0)
            metrics.full_avg300 = values.get("avg300", 0.0)

    return metrics


def get_memory_pressure_sync() -> tuple[MemoryPressureLevel, float, PSIMetrics | None]:
    """Synchronous version of get_memory_pressure for use in non-async contexts.

    Returns:
        A tuple of (pressure_level, free_pct, psi_metrics)
    """
    system = platform.system().lower()

    if system == "darwin":
        level, free_pct = _get_macos_memory_pressure_sync()
        return level, free_pct, None
    elif system == "linux":
        psi = _get_linux_psi_metrics_sync()
        if psi is not None:
            level = psi.to_pressure_level()
            free_pct = max(0.0, 100.0 - psi.some_avg10)
            return level, free_pct, psi
        return MemoryPressureLevel.NORMAL, 100.0, None
    else:
        return MemoryPressureLevel.NORMAL, 100.0, None


def _get_macos_memory_pressure_sync() -> tuple[MemoryPressureLevel, float]:
    """Synchronous macOS memory pressure detection."""
    level = MemoryPressureLevel.NORMAL
    free_pct = 100.0

    # Method 1: sysctl
    try:
        result = subprocess.run(
            ["sysctl", "-n", "vm.memory_pressure"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            value = int(result.stdout.strip())
            if value == 4:
                level = MemoryPressureLevel.CRITICAL
            elif value == 2:
                level = MemoryPressureLevel.WARN
            elif value >= 1:
                level = MemoryPressureLevel.NORMAL
    except (subprocess.SubprocessError, ValueError, subprocess.TimeoutExpired):
        pass

    # Method 2: memory_pressure command
    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            match = re.search(r"free percentage:\s*(\d+)%", result.stdout)
            if match:
                free_pct = float(match.group(1))
                if level == MemoryPressureLevel.NORMAL and free_pct < 20:
                    level = MemoryPressureLevel.WARN
                if free_pct < 10:
                    level = MemoryPressureLevel.CRITICAL
    except (subprocess.SubprocessError, ValueError, subprocess.TimeoutExpired):
        pass

    return level, free_pct


def _get_linux_psi_metrics_sync() -> PSIMetrics | None:
    """Synchronous Linux PSI reading."""
    psi_path = Path("/proc/pressure/memory")

    if not psi_path.exists():
        return None

    try:
        content = psi_path.read_text()
        return _parse_psi_content(content)
    except (OSError, PermissionError):
        return None
