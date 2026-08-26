"""CUDA compatibility patches for MLX on Linux.

MLX on Linux CUDA has some API differences from macOS Metal.
This module provides shims to bridge those gaps.
"""

import sys

import mlx.core as mx


def apply_cuda_compat_patches() -> None:
    """Apply MLX CUDA compatibility patches.

    These patches are only applied on Linux systems where MLX uses the CUDA backend.
    They are no-ops on macOS or CPU-only Linux.
    """
    if sys.platform == "darwin":
        return

    # mlx-lm expects new_thread_local_stream, but Linux CUDA MLX exposes new_stream.
    # Patch mx to provide the expected API.
    if not hasattr(mx, "new_thread_local_stream") and hasattr(mx, "new_stream"):
        mx.new_thread_local_stream = mx.new_stream  # type: ignore[attr-defined]
