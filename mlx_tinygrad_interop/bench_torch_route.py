"""Benchmark tinygrad <-> PyTorch <-> MLX bridge routes.

This file intentionally uses pre-existing interop surfaces instead of adding
new framework patches:

- MLX <-> PyTorch bridge shape based on:
  https://github.com/vllm-project/vllm-metal/blob/main/vllm_metal/pytorch_backend/tensor_bridge.py
- PyTorch -> tinygrad via tinygrad's documented Tensor.from_blob runtime interop

The timed loop excludes source tensor construction and explicit pre-sync, but
still includes the per-call helper and wrapper overhead of the route.
"""

from __future__ import annotations

import argparse
import gc
import platform
import statistics
import sys
import time
from typing import Any, Callable, Literal, cast

import mlx.core as mx
import numpy as np
import torch
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.dtype import DType, _from_torch_dtype, _to_torch_dtype

blackhole: Any = None

_MPS_SAFE_SIZE_BYTES = 1 << 30

DTYPES: dict[str, tuple[Any, Any, np.dtype[Any]]] = {
  "float16": (mx.float16, dtypes.float16, np.dtype(np.float16)),
  "float32": (mx.float32, dtypes.float32, np.dtype(np.float32)),
  "int32": (mx.int32, dtypes.int32, np.dtype(np.int32)),
  "uint8": (mx.uint8, dtypes.uint8, np.dtype(np.uint8)),
}

MLX_TO_TORCH_DTYPE: dict[mx.Dtype, torch.dtype] = {
  mx.float32: torch.float32,
  mx.float16: torch.float16,
  mx.bfloat16: torch.bfloat16,
  mx.int32: torch.int32,
  mx.int64: torch.int64,
  mx.int16: torch.int16,
  mx.int8: torch.int8,
  mx.uint8: torch.uint8,
  mx.bool_: torch.bool,
}

TORCH_TO_MLX_DTYPE: dict[torch.dtype, mx.Dtype] = {v: k for k, v in MLX_TO_TORCH_DTYPE.items()}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Benchmark tinygrad <-> PyTorch <-> MLX bridge overhead.")
  parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
  parser.add_argument("--sizes", default="256,512,1024,2048,4096,7168,8192,16384,32768,65536",
                      help="Comma-separated tensor sizes in bytes.")
  parser.add_argument("--warmup", type=int, default=64)
  parser.add_argument("--samples", type=int, default=8)
  parser.add_argument("--min-batch-us", type=float, default=2000.0)
  parser.add_argument("--torch-device", choices=("cpu", "mps", "auto"), default="auto",
                      help="Intermediate torch device to use for the route.")
  return parser.parse_args()


def bytes_view(mv: memoryview) -> memoryview:
  return mv if mv.format == "B" and mv.ndim == 1 else mv.cast("B")


def get_torch_device(kind: Literal["cpu", "mps", "auto"] = "auto") -> torch.device:
  if kind == "cpu":
    return torch.device("cpu")
  if kind == "mps":
    if not torch.backends.mps.is_available():
      raise RuntimeError("torch MPS backend is not available")
    return torch.device("mps")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def _get_tensor_size_bytes(shape: tuple[int, ...], dtype_itemsize: int) -> int:
  size = dtype_itemsize
  for dim in shape:
    size *= dim
  return size


def sync_mlx() -> None:
  try:
    mx.synchronize()
  except (AttributeError, TypeError):
    mx.eval(mx.array(0, dtype=mx.int32))


def sync_tinygrad() -> None:
  Device["METAL"].synchronize()


def sync_torch(device: torch.device) -> None:
  if device.type == "mps":
    torch.mps.synchronize()
  elif device.type == "cuda":
    torch.cuda.synchronize()


def tinygrad_zero_copy_memoryview(t: Tensor) -> memoryview:
  assert t.device == "METAL", f"expected METAL tensor, got {t.device}"
  buf = cast(Buffer, t.uop.buffer).ensure_allocated()
  return bytes_view(buf.as_memoryview(force_zero_copy=True))


def tinygrad_to_torch(tensor: Tensor, *, device: torch.device | Literal["cpu", "mps"] | None = None,
                      already_contiguous: bool = False) -> torch.Tensor:
  if device is None:
    device = get_torch_device("auto")
  elif isinstance(device, str):
    device = torch.device(device)

  if not already_contiguous:
    tensor = tensor.contiguous()
  tensor = tensor.cast(tensor.dtype.base).realize()

  torch_dtype = _to_torch_dtype(tensor.dtype.base)
  if torch_dtype is None:
    raise ValueError(f"Unsupported tinygrad dtype: {tensor.dtype}")

  if tensor.device == "METAL":
    sync_tinygrad()
    buffer = tinygrad_zero_copy_memoryview(tensor)
  else:
    if tensor.device != "CPU":
      tensor = tensor.to("CPU").realize()
    buffer = bytes_view(tensor.data())

  out = torch.frombuffer(buffer, dtype=torch_dtype).reshape(tuple(int(dim) for dim in tensor.shape))

  if device.type == "mps":
    if _get_tensor_size_bytes(tuple(int(dim) for dim in tensor.shape), tensor.dtype.itemsize) < _MPS_SAFE_SIZE_BYTES:
      out = out.to(device)
  elif device.type != "cpu":
    out = out.to(device)

  return out


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
  if tensor.device.type != "cpu":
    sync_torch(tensor.device)
    tensor = tensor.cpu()
  tensor = tensor.detach()
  if tensor.dtype == torch.bfloat16:
    return mx.array(tensor)
  return mx.array(tensor.numpy())


def mlx_to_torch(array: mx.array, *, device: torch.device | Literal["cpu", "mps"] | None = None,
                 already_contiguous: bool = False) -> torch.Tensor:
  if device is None:
    device = get_torch_device("auto")
  elif isinstance(device, str):
    device = torch.device(device)

  torch_dtype = MLX_TO_TORCH_DTYPE.get(array.dtype)
  if torch_dtype is None:
    raise ValueError(f"Unsupported MLX dtype: {array.dtype}")

  if not already_contiguous:
    array = mx.contiguous(array)
  mx.eval(array)
  out = torch.frombuffer(memoryview(array), dtype=torch_dtype).reshape(tuple(int(dim) for dim in array.shape))

  if device.type == "mps":
    if _get_tensor_size_bytes(tuple(int(dim) for dim in array.shape), int(array.dtype.size)) < _MPS_SAFE_SIZE_BYTES:
      out = out.to(device)
  elif device.type != "cpu":
    out = out.to(device)

  return out


def torch_to_tinygrad(tensor: torch.Tensor) -> Tensor:
  tensor = tensor.detach()
  if not tensor.is_contiguous():
    tensor = tensor.contiguous()

  if tensor.device.type == "mps":
    sync_torch(tensor.device)
    target_device = "METAL"
  elif tensor.device.type == "cuda":
    sync_torch(tensor.device)
    target_device = "CUDA"
  elif tensor.device.type == "cpu":
    target_device = "CPU"
  else:
    raise ValueError(f"Unsupported torch device: {tensor.device}")

  out = Tensor.from_blob(
    tensor.data_ptr(),
    tuple(int(dim) for dim in tensor.shape),
    dtype=_from_torch_dtype(tensor.dtype),
    device=target_device,
  )
  if out.uop.has_buffer_identity():
    setattr(cast(Buffer, out.uop.buffer).base, "_external_owner", tensor)
  return out


def tinygrad_to_mlx_via_torch(tensor: Tensor, *, torch_device: torch.device) -> mx.array:
  return torch_to_mlx(tinygrad_to_torch(tensor, device=torch_device))


def mlx_to_tinygrad_via_torch(array: mx.array, *, torch_device: torch.device) -> Tensor:
  return torch_to_tinygrad(mlx_to_torch(array, device=torch_device))


def tinygrad_to_mlx_direct(tensor: Tensor) -> mx.array:
  return mx.array(tinygrad_zero_copy_memoryview(tensor))


def mlx_to_tinygrad_direct(array: mx.array, tg_dtype: DType) -> Tensor:
  storage = mx.metal._unsafe_export_storage(array)
  return Tensor._unsafe_from_metal_buffer_fast(
    int(storage["mtl_buffer_ptr"]),
    tuple(storage["shape"]),
    dtype=tg_dtype,
    byte_offset=int(storage["offset_bytes"]),
    buffer_nbytes=int(storage["buffer_nbytes"]),
    owner=array,
  )


def bench_callable(fn: Callable[[], Any], warmup: int, samples: int, min_batch_us: float) -> dict[str, float]:
  global blackhole
  for _ in range(warmup):
    blackhole = fn()

  min_batch_ns = int(min_batch_us * 1000.0)
  iters = 1
  while True:
    start = time.perf_counter_ns()
    for _ in range(iters):
      blackhole = fn()
    elapsed = time.perf_counter_ns() - start
    if elapsed >= min_batch_ns or iters >= (1 << 20):
      break
    iters *= 2

  vals_us: list[float] = []
  for _ in range(samples):
    start = time.perf_counter_ns()
    for _ in range(iters):
      blackhole = fn()
    elapsed = time.perf_counter_ns() - start
    vals_us.append(elapsed / iters / 1000.0)

  return {
    "iters": float(iters),
    "min_us": min(vals_us),
    "median_us": statistics.median(vals_us),
    "mean_us": statistics.mean(vals_us),
    "stdev_us": statistics.stdev(vals_us) if len(vals_us) > 1 else 0.0,
  }


def assert_equal(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
  if np.issubdtype(expected.dtype, np.floating):
    np.testing.assert_allclose(actual, expected, rtol=5e-5 if expected.dtype == np.float32 else 5e-3, atol=1e-5 if expected.dtype == np.float32 else 5e-3, err_msg=name)
  else:
    np.testing.assert_array_equal(actual, expected, err_msg=name)


def print_header(dtype_name: str, torch_device: torch.device) -> None:
  print(f"# python={platform.python_version()} platform={platform.platform()}")
  print(f"# dtype={dtype_name} mlx_metal_available={mx.metal.is_available()} tinygrad_device=METAL torch_device={torch_device}")
  print("# sizes are source tensor sizes in bytes")
  print("# timed loop excludes source tensor construction and explicit pre-sync, but still includes helper and wrapper overhead")
  print("# via_torch_route rows use pre-existing tinygrad<->torch and mlx<->torch bridges without framework patches")
  print("size_bytes,method,direction,min_us,median_us,mean_us,stdev_us,iters")


def main() -> None:
  args = parse_args()
  if not mx.metal.is_available():
    raise RuntimeError("MLX Metal is not available")
  torch_device = get_torch_device(cast(Literal["cpu", "mps", "auto"], args.torch_device))
  mx_dtype, tg_dtype, np_dtype = DTYPES[args.dtype]
  sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

  gc.disable()
  try:
    print_header(args.dtype, torch_device)
    for size_bytes in sizes:
      if size_bytes <= 0:
        continue
      if size_bytes % np_dtype.itemsize != 0:
        print(f"# skipping size {size_bytes}: not divisible by dtype itemsize {np_dtype.itemsize}", file=sys.stderr)
        continue

      numel = size_bytes // np_dtype.itemsize
      values = np.arange(numel, dtype=np_dtype)
      src_tg = Tensor(values, device="METAL", dtype=tg_dtype).realize()
      src_mx = mx.array(values, dtype=mx_dtype)
      sync_tinygrad()
      sync_mlx()

      # correctness checks stay outside the timed loop
      assert_equal("tinygrad->mlx via torch raw", np.array(tinygrad_to_mlx_via_torch(src_tg, torch_device=torch_device)), values)
      assert_equal("mlx->tinygrad via torch raw", mlx_to_tinygrad_via_torch(src_mx, torch_device=torch_device).numpy(), values)

      benches: list[tuple[str, str, Callable[[], Any]]] = [
        ("via_torch_route", "tinygrad_to_mlx", lambda s=src_tg, td=torch_device: tinygrad_to_mlx_via_torch(s, torch_device=td)),
        ("via_torch_route", "mlx_to_tinygrad", lambda s=src_mx, td=torch_device: mlx_to_tinygrad_via_torch(s, torch_device=td)),
        ("bridge_half", "tinygrad_to_torch", lambda s=src_tg, td=torch_device: tinygrad_to_torch(s, device=td)),
        ("bridge_half", "torch_to_mlx", lambda s=src_tg, td=torch_device: torch_to_mlx(tinygrad_to_torch(s, device=td))),
        ("bridge_half", "mlx_to_torch", lambda s=src_mx, td=torch_device: mlx_to_torch(s, device=td)),
        ("bridge_half", "torch_to_tinygrad", lambda s=src_mx, td=torch_device: torch_to_tinygrad(mlx_to_torch(s, device=td))),
        ("direct_baseline", "tinygrad_to_mlx", lambda s=src_tg: tinygrad_to_mlx_direct(s)),
        ("direct_baseline", "mlx_to_tinygrad", lambda s=src_mx, td=tg_dtype: mlx_to_tinygrad_direct(s, td)),
      ]

      for method, direction, fn in benches:
        try:
          blackhole = fn()
        except Exception as exc:
          print(f"# skipping {size_bytes},{method},{direction}: {exc}", file=sys.stderr)
          continue
        stats = bench_callable(fn, warmup=args.warmup, samples=args.samples, min_batch_us=args.min_batch_us)
        print(
          f"{size_bytes},{method},{direction},"
          f"{stats['min_us']:.3f},{stats['median_us']:.3f},{stats['mean_us']:.3f},{stats['stdev_us']:.3f},{int(stats['iters'])}"
        )
  finally:
    gc.enable()


if __name__ == "__main__":
  main()
