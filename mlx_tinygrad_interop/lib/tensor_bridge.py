"""Tensor bridge between tinygrad and MLX.

Baseline implementation modelled after:
https://github.com/vllm-project/vllm-metal/blob/main/vllm_metal/pytorch_backend/tensor_bridge.py

This module currently implements the `tinygrad -> MLX` direction only.
`MLX -> tinygrad` is left stubbed on purpose.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

import mlx.core as mx
from tinygrad.tensor import Tensor
from tinygrad.device import Buffer, Device
from tinygrad.dtype import DType, dtypes

logger = logging.getLogger(__name__)

# MPS has a 4GB (2^32 bytes) limit for MPSTemporaryNDArray allocations.
# Metal may allocate multiple temporary buffers internally, so we use a
# conservative threshold of 1GB to avoid hitting the limit.
# See: https://github.com/anthropics/vllm-metal/issues/43
_MPS_SAFE_SIZE_BYTES = 1 << 30  # 1GB

# MLX to tinygrad dtype mapping
MLX_TO_TINYGRAD_DTYPE: dict[mx.Dtype, DType] = {
    mx.float32: dtypes.float32,
    mx.float16: dtypes.float16,
    mx.bfloat16: dtypes.bfloat16,
    mx.int32: dtypes.int32,
    mx.int64: dtypes.int64,
    mx.int16: dtypes.int16,
    mx.int8: dtypes.int8,
    mx.uint8: dtypes.uint8,
    mx.bool_: dtypes.bool,
}

# tinygrad to MLX dtype mapping
TINYGRAD_TO_MLX_DTYPE: dict[DType, mx.Dtype] = {
  v: k for k, v in MLX_TO_TINYGRAD_DTYPE.items()
}


def _get_tensor_size_bytes(tensor: Tensor) -> int:
  """Calculate the size of a tinygrad tensor in bytes."""
  return tensor.numel() * tensor.dtype.itemsize


def _get_buffer_view(tensor: Tensor, *, already_contiguous: bool = False) -> memoryview:
  """Expose a tinygrad tensor as a Python buffer.

  For METAL tensors on Apple Silicon, this uses tinygrad's zero-copy
  `as_memoryview(force_zero_copy=True)` path after forcing realization and a
  Metal synchronize. For other devices, it falls back to the standard
  `Tensor.data()` path.
  """
  tensor = tensor.cast(tensor.dtype.base)
  if not already_contiguous:
    tensor = tensor.contiguous()

  if tensor.device == "METAL":
    tensor = tensor.realize()
    sync_tinygrad()
    if tensor.dtype.base.fmt is None:
      raise ValueError(f"Unsupported tinygrad dtype for memoryview bridge: {tensor.dtype}")
    buf = cast(Buffer, tensor.uop.buffer).ensure_allocated()
    return buf.as_memoryview(force_zero_copy=True).cast(tensor.dtype.base.fmt, tensor.shape)

  if tensor.device != "CPU":
    tensor = tensor.to("CPU").realize()

  return tensor.data()


def tinygrad_to_mlx(tensor: Tensor, *, already_contiguous: bool = False) -> mx.array:
  """Convert a tinygrad tensor to an MLX array.

  Uses a buffer-protocol / memoryview path when possible. In current MLX this
  still creates a fresh MLX array rather than aliasing the tinygrad buffer, but
  it is the closest analogue to the reference vLLM bridge's public shape.

  Args:
    tensor: tinygrad tensor
    already_contiguous: Skip the contiguity step if the tensor is already known
      to be dense row-major contiguous.

  Returns:
    MLX array with the same logical values.
  """
  if tensor.dtype.base not in TINYGRAD_TO_MLX_DTYPE:
    raise ValueError(f"Unsupported tinygrad dtype: {tensor.dtype}")

  buffer = _get_buffer_view(tensor, already_contiguous=already_contiguous)
  array = mx.array(buffer)
  if array.dtype != TINYGRAD_TO_MLX_DTYPE[tensor.dtype.base]:
    array = array.astype(TINYGRAD_TO_MLX_DTYPE[tensor.dtype.base])
  return array


def mlx_to_tinygrad(array: mx.array) -> Tensor:
  """Convert an MLX array to a tinygrad tensor.

  This direction is intentionally left stubbed here. The current repo already
  carries more specialized MLX -> tinygrad experiments in the lease-pool and
  benchmark helpers, and this baseline bridge module is only meant to mirror
  the public shape of the vLLM bridge for `tinygrad -> MLX`.
  """
  raise NotImplementedError("mlx_to_tinygrad() is intentionally stubbed in this baseline bridge")


def sync_mlx() -> None:
  """Synchronize MLX operations."""
  try:
    mx.synchronize()
  except (AttributeError, TypeError):
    mx.eval(mx.array(0, dtype=mx.int32))


def sync_tinygrad() -> None:
  """Synchronize tinygrad METAL operations."""
  try:
    Device["METAL"].synchronize()
  except Exception:
    logger.debug("tinygrad METAL synchronize unavailable", exc_info=True)

