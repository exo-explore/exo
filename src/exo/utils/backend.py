"""Runtime GPU backend detection for exo.

Call :func:`detect_backend` once at process start (before any ML framework
imports) to determine which inference engine to use.  The result is cached
so subsequent calls are free.
"""

import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Literal

BackendName = Literal["mlx_metal", "mlx_cuda", "rocm", "vulkan", "cpu"]


def _probe_cuda() -> bool:
    """Return True when libcuda.so.1 can be loaded (NVIDIA GPU + driver present)."""
    try:
        import ctypes

        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


def _probe_rocm() -> bool:
    """Return True when ROCm is installed and a ROCm-capable device is visible."""
    import os

    if not os.path.isdir("/opt/rocm"):
        return False
    rocm_smi = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"
    try:
        result = subprocess.run(
            [rocm_smi, "--showid"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _probe_vulkan() -> bool:
    """Return True when vulkaninfo can enumerate at least one physical device."""
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo is None:
        return False
    try:
        result = subprocess.run(
            [vulkaninfo, "--summary"],
            capture_output=True,
            timeout=5,
        )
        # vulkaninfo exits 0 and prints GPU info when a Vulkan device is found
        return result.returncode == 0 and b"GPU" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


@lru_cache(maxsize=1)
def detect_backend() -> BackendName:
    """Detect the best available GPU backend for this machine.

    Detection order (first match wins):
    1. ``mlx_metal``  – macOS with Apple Silicon (MLX Metal backend)
    2. ``mlx_cuda``   – Linux + NVIDIA GPU (MLX CUDA backend)
    3. ``rocm``       – Linux + AMD GPU with ROCm driver stack
    4. ``vulkan``     – Linux + any GPU with Vulkan support
    5. ``cpu``        – fallback: software only

    The function is cached so it runs at most once per process.
    """
    if sys.platform == "darwin":
        # On macOS we always use MLX; Metal availability is checked inside mlx
        return "mlx_metal"

    # Linux paths only below
    if _probe_cuda():
        return "mlx_cuda"

    if _probe_rocm():
        return "rocm"

    if _probe_vulkan():
        return "vulkan"

    return "cpu"
