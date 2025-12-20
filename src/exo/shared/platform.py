"""
Platform detection utilities for exo.

Determines which inference backend to use based on the platform.
Includes ARM-specific feature detection for optimization on Android/Termux.
"""

import os
import platform
import sys
from typing import Literal, TypedDict

from exo.shared.types.worker.instances import InstanceMeta

BackendType = Literal["mlx", "llamacpp", "unknown"]
DeviceTier = Literal["low", "medium", "high", "ultra"]


class ARMFeatures(TypedDict, total=False):
    """ARM CPU features detected from /proc/cpuinfo."""

    neon: bool
    dotprod: bool
    fp16: bool
    sve: bool
    sve2: bool
    i8mm: bool
    bf16: bool


class ARMInfo(TypedDict, total=False):
    """ARM-specific platform information."""

    features: ARMFeatures
    ram_gb: int
    tier: DeviceTier
    core_count: int
    recommended_threads: int


def get_platform_info() -> dict[str, str]:
    """Get detailed platform information."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "sys_platform": sys.platform,
    }


def get_arm_features() -> ARMFeatures:
    """
    Detect ARM CPU features from /proc/cpuinfo.

    Returns a dictionary of available ARM extensions that are relevant
    for optimizing LLM inference performance.

    Key extensions for LLM inference:
    - dotprod: 2-4x speedup for quantized models
    - fp16: 2x throughput for half-precision
    - i8mm: 2-4x speedup for int8 models
    - sve2: Scalable vectors for ML workloads
    """
    features: ARMFeatures = {
        "neon": False,
        "dotprod": False,
        "fp16": False,
        "sve": False,
        "sve2": False,
        "i8mm": False,
        "bf16": False,
    }

    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpuinfo = f.read().lower()

            # Check for features in the Features line
            for line in cpuinfo.split("\n"):
                if "features" in line:
                    feature_str = line.split(":")[-1] if ":" in line else ""

                    # NEON/ASIMD (always present on AArch64)
                    if "asimd" in feature_str or "neon" in feature_str:
                        features["neon"] = True

                    # Dot product (huge for quantized inference)
                    if "asimddp" in feature_str or "dotprod" in feature_str:
                        features["dotprod"] = True

                    # FP16 support
                    if "fphp" in feature_str or "asimdhp" in feature_str:
                        features["fp16"] = True

                    # SVE (Scalable Vector Extension)
                    if "sve2" in feature_str:
                        features["sve2"] = True
                        features["sve"] = True
                    elif "sve" in feature_str:
                        features["sve"] = True

                    # Int8 Matrix Multiply (ARMv8.6+)
                    if "i8mm" in feature_str:
                        features["i8mm"] = True

                    # BFloat16
                    if "bf16" in feature_str:
                        features["bf16"] = True

                    break

    except (OSError, IOError):
        pass

    return features


def get_memory_gb() -> int:
    """Get total system RAM in gigabytes."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    # MemTotal is in kB
                    kb = int(line.split()[1])
                    return kb // (1024 * 1024)
    except (OSError, IOError, ValueError):
        pass

    # Fallback using psutil if available
    try:
        import psutil

        return psutil.virtual_memory().total // (1024 * 1024 * 1024)
    except ImportError:
        pass

    return 4  # Safe default


def get_device_tier() -> DeviceTier:
    """
    Classify device into performance tiers based on RAM.

    Tiers determine recommended model sizes:
    - low (≤4GB): 0.5B-1.5B models
    - medium (5-6GB): 1.5B-3B models
    - high (7-8GB): 3B-7B models
    - ultra (>8GB): 7B-9B models (single device)
    """
    ram_gb = get_memory_gb()

    if ram_gb <= 4:
        return "low"
    elif ram_gb <= 6:
        return "medium"
    elif ram_gb <= 8:
        return "high"
    else:
        return "ultra"


def get_core_count() -> int:
    """Get the number of CPU cores."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def get_recommended_thread_count() -> int:
    """
    Get recommended thread count for LLM inference.

    On ARM big.LITTLE architectures, we typically want to use
    only the big cores (usually 4) for compute-intensive tasks.
    """
    total_cores = get_core_count()

    if is_android():
        # On Android, big.LITTLE typically has 4 big + 4 little cores
        # Use big cores only (usually half)
        return min(4, total_cores // 2) if total_cores > 4 else total_cores
    else:
        # On other platforms, use all cores minus 1
        return max(1, total_cores - 1)


def get_arm_info() -> ARMInfo:
    """Get comprehensive ARM platform information."""
    features = get_arm_features()
    ram_gb = get_memory_gb()
    tier = get_device_tier()
    core_count = get_core_count()
    threads = get_recommended_thread_count()

    return {
        "features": features,
        "ram_gb": ram_gb,
        "tier": tier,
        "core_count": core_count,
        "recommended_threads": threads,
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

