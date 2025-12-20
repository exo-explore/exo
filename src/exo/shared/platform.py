"""
Platform detection utilities for exo.

Determines which inference backend to use based on the platform.
"""

import platform
import sys
from typing import Literal

from exo.shared.types.worker.instances import InstanceMeta

BackendType = Literal["mlx", "llamacpp", "unknown"]


def get_platform_info() -> dict[str, str]:
    """Get detailed platform information."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "sys_platform": sys.platform,
    }


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3/etc)."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system != "darwin":
        return False

    return "arm" in machine or "aarch64" in machine


def is_android() -> bool:
    """
    Check if running on Android (Termux).
    Android/Termux reports as Linux but has specific characteristics.
    """
    if sys.platform != "linux":
        return False

    # Check for Android-specific paths
    import os

    android_indicators = [
        "/data/data/com.termux",
        os.environ.get("TERMUX_VERSION"),
        os.environ.get("ANDROID_ROOT"),
    ]

    return any(indicator for indicator in android_indicators if indicator)


def is_linux() -> bool:
    """Check if running on Linux (excluding Android)."""
    return sys.platform == "linux" and not is_android()


def get_recommended_backend() -> BackendType:
    """
    Get the recommended inference backend for this platform.

    Returns:
        - "mlx" for Apple Silicon Macs
        - "llamacpp" for Android, Linux, and other platforms
        - "unknown" if unable to determine
    """
    if is_apple_silicon():
        return "mlx"

    if is_android() or is_linux():
        return "llamacpp"

    # Windows or other platforms
    if sys.platform == "win32":
        return "llamacpp"

    return "unknown"


def get_recommended_instance_meta() -> InstanceMeta:
    """
    Get the recommended InstanceMeta type for this platform.

    Returns the appropriate InstanceMeta enum value based on platform.
    """
    backend = get_recommended_backend()

    if backend == "mlx":
        return InstanceMeta.MlxRing

    return InstanceMeta.LlamaCpp


def check_backend_available(backend: BackendType) -> bool:
    """
    Check if a specific backend is available (importable) on this platform.
    """
    if backend == "mlx":
        try:
            import mlx  # noqa: F401

            return True
        except ImportError:
            return False

    if backend == "llamacpp":
        try:
            import llama_cpp  # noqa: F401

            return True
        except ImportError:
            return False

    return False


def get_available_backends() -> list[BackendType]:
    """Get list of available backends on this platform."""
    available: list[BackendType] = []

    if check_backend_available("mlx"):
        available.append("mlx")

    if check_backend_available("llamacpp"):
        available.append("llamacpp")

    return available

