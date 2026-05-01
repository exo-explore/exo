import argparse
import gc
import platform
import statistics
import sys
import time
from typing import Any, Callable, cast

import mlx.core as mx
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer

try:
  from mlx_tinygrad_interop.lib.lease_pool import (
    MlxToTinygradCopyLeasePool,
    MlxToTinygradCopyLeasePools,
    MlxToTinygradLeasePool,
    MlxToTinygradLeasePools,
  )
except ModuleNotFoundError:
  from lib.lease_pool import MlxToTinygradCopyLeasePool, MlxToTinygradCopyLeasePools, MlxToTinygradLeasePool, MlxToTinygradLeasePools

blackhole: Any = None

DTYPES: dict[str, tuple[Any, Any, np.dtype[Any]]] = {
  "float16": (mx.float16, dtypes.float16, np.dtype(np.float16)),
  "float32": (mx.float32, dtypes.float32, np.dtype(np.float32)),
  "int32": (mx.int32, dtypes.int32, np.dtype(np.int32)),
  "uint8": (mx.uint8, dtypes.uint8, np.dtype(np.uint8)),
}


class Alternator:
  def __init__(self, *items: Any):
    assert items, "Alternator needs at least one item"
    self.items = items
    self.index = 0

  def next(self) -> Any:
    item = self.items[self.index]
    self.index = (self.index + 1) % len(self.items)
    return item


class BorrowerRing:
  def __init__(self, *borrowers: Any):
    assert borrowers, "BorrowerRing needs at least one borrower"
    self.borrowers = borrowers
    self.index = 0

  def next(self) -> Any:
    borrower = self.borrowers[self.index]
    self.index = (self.index + 1) % len(self.borrowers)
    return borrower


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Benchmark raw tinygrad <-> MLX tensor conversion overhead.")
  parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
  parser.add_argument("--sizes", default="256,512,1024,2048,4096,7168,8192,16384,32768,65536,262144,1048576",
                      help="Comma-separated tensor sizes in bytes.")
  parser.add_argument("--warmup", type=int, default=128)
  parser.add_argument("--samples", type=int, default=12)
  parser.add_argument("--min-batch-us", type=float, default=2000.0,
                      help="Minimum target batch duration per sample in microseconds.")
  return parser.parse_args()


def bytes_view(mv: memoryview) -> memoryview:
  return mv if mv.format == "B" and mv.ndim == 1 else mv.cast("B")


def mlx_dtype_name(dtype: Any) -> str:
  return repr(dtype).removeprefix("mlx.core.")


def tinygrad_zero_copy_memoryview(t: Tensor) -> memoryview:
  assert t.device == "METAL", f"expected METAL tensor, got {t.device}"
  buf = cast(Buffer, t.uop.buffer).ensure_allocated()
  assert t.dtype.base.fmt is not None, f"no buffer format for dtype {t.dtype.base}"
  return buf.as_memoryview(force_zero_copy=True).cast(t.dtype.base.fmt, t.shape)


def tinygrad_from_mlx_legacy(x: Any, tg_dtype: Any) -> Tensor:
  storage = mx.metal._unsafe_export_storage(x)
  return Tensor._unsafe_from_metal_buffer(
    int(storage["mtl_buffer_ptr"]),
    tuple(storage["shape"]),
    dtype=tg_dtype,
    byte_offset=int(storage["offset_bytes"]),
    buffer_nbytes=int(storage["buffer_nbytes"]),
    owner=x,
  )


def tinygrad_from_mlx_fast(x: Any, tg_dtype: Any) -> Tensor:
  storage = mx.metal._unsafe_export_storage(x)
  return Tensor._unsafe_from_metal_buffer_fast(
    int(storage["mtl_buffer_ptr"]),
    tuple(storage["shape"]),
    dtype=tg_dtype,
    byte_offset=int(storage["offset_bytes"]),
    buffer_nbytes=int(storage["buffer_nbytes"]),
    owner=x,
  )


def tinygrad_from_mlx_single_entry(x: Any, tg_dtype: Any) -> Tensor:
  return mx.metal._unsafe_to_tinygrad_fast(x, tg_dtype, owner=x)


def tinygrad_from_mlx_reuse(x: Any, borrower: Any) -> Tensor:
  return mx.metal._unsafe_rebind_tinygrad(x, borrower, owner=x)


def tinygrad_from_mlx_lease_acquire_release(x: Any, pool: MlxToTinygradLeasePool) -> int:
  lease = pool.acquire_from_mlx(x)
  generation = lease.generation
  lease.release(synchronize=False)
  return generation


def tinygrad_from_mlx_lease_then_use_sum(x: Any, pool: MlxToTinygradLeasePool) -> Tensor:
  lease = pool.acquire_from_mlx(x)
  try:
    return tinygrad_consume_sum(lease.tensor)
  finally:
    lease.release(synchronize=False)


def tinygrad_from_mlx_scoped_handoff_noop(x: Any, pools: MlxToTinygradLeasePools, tg_dtype: Any) -> int:
  return pools.run_with_mlx_tensor(x, tg_dtype=tg_dtype, fn=lambda _: 0)


def tinygrad_from_mlx_scoped_handoff_then_use_sum(x: Any, pools: MlxToTinygradLeasePools, tg_dtype: Any) -> Tensor:
  return pools.run_with_mlx_tensor(x, tg_dtype=tg_dtype, fn=lambda t: tinygrad_consume_sum(t))


def tinygrad_from_mlx_copy_pool_acquire_release(x: Any, pool: MlxToTinygradCopyLeasePool) -> int:
  lease = pool.acquire_from_mlx(x)
  generation = lease.generation
  lease.release(synchronize=False)
  return generation


def tinygrad_from_mlx_copy_pool_then_use_sum(x: Any, pool: MlxToTinygradCopyLeasePool) -> Tensor:
  lease = pool.acquire_from_mlx(x)
  try:
    return tinygrad_consume_sum(lease.tensor)
  finally:
    lease.release(synchronize=False)


def tinygrad_from_mlx_scoped_copy_handoff_noop(x: Any, pools: MlxToTinygradCopyLeasePools, tg_dtype: Any) -> int:
  return pools.run_with_mlx_tensor(x, tg_dtype=tg_dtype, fn=lambda _: 0)


def tinygrad_from_mlx_scoped_copy_handoff_then_use_sum(x: Any, pools: MlxToTinygradCopyLeasePools, tg_dtype: Any) -> Tensor:
  return pools.run_with_mlx_tensor(x, tg_dtype=tg_dtype, fn=lambda t: tinygrad_consume_sum(t))


def tinygrad_consume_sum(t: Tensor) -> Tensor:
  out = (t + 1).sum()
  out.realize()
  Device["METAL"].synchronize()
  return out


def mlx_from_tinygrad_maybe_copy(t: Tensor, mx_dtype: Any) -> Any:
  storage = t._unsafe_metal_storage()
  return mx.metal._unsafe_array_from_ptr(
    int(storage["raw_ptr"]),
    tuple(storage["shape"]),
    mx_dtype,
    owner=t,
  )


def mlx_from_tinygrad_alias_only(t: Tensor, mx_dtype: Any) -> Any:
  storage = t._unsafe_metal_storage()
  return mx.metal._unsafe_array_from_ptr_alias_only(
    int(storage["raw_ptr"]),
    tuple(storage["shape"]),
    mx_dtype,
    owner=t,
  )


def tinygrad_from_mlx_copy(x: Any, tg_dtype: Any) -> Tensor:
  out = Tensor.empty(*tuple(int(dim) for dim in x.shape), device="METAL", dtype=tg_dtype)
  cast(Buffer, out.uop.buffer).ensure_allocated().copyin(bytes_view(memoryview(x)))
  return out


def mlx_from_tinygrad_copy(t: Tensor) -> Any:
  return mx.array(tinygrad_zero_copy_memoryview(t))


def tinygrad_from_mlx_numpy(x: Any) -> Tensor:
  out = Tensor(np.array(x, copy=True), device="METAL")
  out.realize()
  Device["METAL"].synchronize()
  return out


def mlx_from_tinygrad_numpy(t: Tensor) -> Any:
  return mx.array(t.numpy())


def mlx_export_storage(x: Any) -> Any:
  return mx.metal._unsafe_export_storage(x)


def tinygrad_import_from_storage(storage: Any, tg_dtype: Any, owner: Any) -> Tensor:
  return Tensor._unsafe_from_metal_buffer(
    int(storage["mtl_buffer_ptr"]),
    tuple(storage["shape"]),
    dtype=tg_dtype,
    byte_offset=int(storage["offset_bytes"]),
    buffer_nbytes=int(storage["buffer_nbytes"]),
    owner=owner,
  )


def tinygrad_import_from_storage_fast(storage: Any, tg_dtype: Any, owner: Any) -> Tensor:
  return Tensor._unsafe_from_metal_buffer_fast(
    int(storage["mtl_buffer_ptr"]),
    tuple(storage["shape"]),
    dtype=tg_dtype,
    byte_offset=int(storage["offset_bytes"]),
    buffer_nbytes=int(storage["buffer_nbytes"]),
    owner=owner,
  )


def tinygrad_import_from_storage_reuse(storage: Any, owner: Any, borrower: Any) -> Tensor:
  return borrower.rebind(
    int(storage["mtl_buffer_ptr"]),
    owner=owner,
    shape=tuple(storage["shape"]),
    dtype_name=mlx_dtype_name(storage["dtype"]),
    byte_offset=int(storage["offset_bytes"]),
    buffer_nbytes=int(storage["buffer_nbytes"]),
  )


def tinygrad_export_storage(t: Tensor) -> Any:
  return t._unsafe_metal_storage()


def mlx_import_from_storage(storage: Any, mx_dtype: Any, owner: Any) -> Any:
  return mx.metal._unsafe_array_from_ptr(
    int(storage["raw_ptr"]),
    tuple(storage["shape"]),
    mx_dtype,
    owner=owner,
  )


def mlx_import_from_storage_alias_only(storage: Any, mx_dtype: Any, owner: Any) -> Any:
  return mx.metal._unsafe_array_from_ptr_alias_only(
    int(storage["raw_ptr"]),
    tuple(storage["shape"]),
    mx_dtype,
    owner=owner,
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
    "avg_us": statistics.mean(vals_us),
    "stddev_us": statistics.stdev(vals_us) if len(vals_us) > 1 else 0.0,
  }


def print_header(dtype_name: str) -> None:
  print(f"# python={platform.python_version()} platform={platform.platform()}")
  print(f"# dtype={dtype_name} mlx_metal_available={mx.metal.is_available()} tinygrad_device=METAL")
  print("# sizes are source tensor sizes in bytes")
  print("# timed loop excludes source tensor construction and explicit pre-sync, but still includes per-call helper, binding, and wrapper overhead")
  print("# reported latency is average per-call time with sample standard deviation")
  print("# rebindable_slot_* rows rebind and return the same tinygrad Tensor object each time; ring rows rotate through multiple such slots")
  print("# copy_pool_* rows reuse tinygrad-owned destination tensors and copy source bytes into them before release")
  print("size_bytes,method,direction,avg_us,stddev_us,iters")


def main() -> None:
  args = parse_args()
  mx_dtype, tg_dtype, np_dtype = DTYPES[args.dtype]
  sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

  required = [
    ("mx.metal._unsafe_export_storage", getattr(mx.metal, "_unsafe_export_storage", None)),
    ("mx.metal._unsafe_to_tinygrad_fast", getattr(mx.metal, "_unsafe_to_tinygrad_fast", None)),
    ("mx.metal._unsafe_rebind_tinygrad", getattr(mx.metal, "_unsafe_rebind_tinygrad", None)),
    ("mx.metal._unsafe_array_from_ptr", getattr(mx.metal, "_unsafe_array_from_ptr", None)),
    ("mx.metal._unsafe_array_from_ptr_alias_only", getattr(mx.metal, "_unsafe_array_from_ptr_alias_only", None)),
    ("Tensor._unsafe_from_metal_buffer", getattr(Tensor, "_unsafe_from_metal_buffer", None)),
    ("Tensor._unsafe_from_metal_buffer_fast", getattr(Tensor, "_unsafe_from_metal_buffer_fast", None)),
    ("Tensor._unsafe_metal_borrower", getattr(Tensor, "_unsafe_metal_borrower", None)),
    ("Tensor._unsafe_metal_storage", getattr(Tensor, "_unsafe_metal_storage", None)),
  ]
  missing = [name for name, value in required if value is None]
  if missing:
    raise RuntimeError(f"Missing required helper(s): {', '.join(missing)}")

  gc.disable()
  try:
    print_header(args.dtype)
    for size_bytes in sizes:
      if size_bytes <= 0:
        continue
      if size_bytes % np_dtype.itemsize != 0:
        print(f"# skipping size {size_bytes}: not divisible by dtype itemsize {np_dtype.itemsize}", file=sys.stderr)
        continue

      numel = size_bytes // np_dtype.itemsize

      # Build one realized source tensor on each side. Source creation and the
      # explicit pre-sync stay outside the timed loop, but per-call helper,
      # binding, owner-pinning, and wrapper construction still remain inside it.
      src_mx_pool = [mx.array(np.arange(numel, dtype=np_dtype) + np.array(i, dtype=np_dtype), dtype=mx_dtype) for i in range(4)]
      src_mx = src_mx_pool[0]

      src_tg = Tensor(np.arange(numel, dtype=np_dtype), device="METAL").realize()
      Device["METAL"].synchronize()

      mx_storage_pool = [(mlx_export_storage(src), src) for src in src_mx_pool]
      mx_storage = mx_storage_pool[0][0]
      tg_storage = tinygrad_export_storage(src_tg)
      mx_slot_borrower = Tensor._unsafe_metal_borrower(
        int(mx_storage["mtl_buffer_ptr"]),
        tuple(mx_storage["shape"]),
        dtype=tg_dtype,
        byte_offset=int(mx_storage["offset_bytes"]),
        buffer_nbytes=int(mx_storage["buffer_nbytes"]),
        owner=src_mx,
      )
      mx_slot_sources = Alternator(*src_mx_pool)
      mx_slot_storage = Alternator(*mx_storage_pool)
      mx_ring = BorrowerRing(*[
        Tensor._unsafe_metal_borrower(
          int(mx_storage["mtl_buffer_ptr"]),
          tuple(mx_storage["shape"]),
          dtype=tg_dtype,
          byte_offset=int(mx_storage["offset_bytes"]),
          buffer_nbytes=int(mx_storage["buffer_nbytes"]),
          owner=src_mx,
        ) for _ in range(4)
      ])
      lease_pool = MlxToTinygradLeasePool.from_mlx(src_mx, tg_dtype=tg_dtype, capacity=4, synchronize_on_release=True)
      scoped_handoff_pools = MlxToTinygradLeasePools(capacity_per_key=4, synchronize_on_release=True)
      copy_lease_pool = MlxToTinygradCopyLeasePool.from_mlx(src_mx, tg_dtype=tg_dtype, capacity=4, synchronize_on_release=True)
      scoped_copy_handoff_pools = MlxToTinygradCopyLeasePools(capacity_per_key=4, synchronize_on_release=True)

      benches: list[tuple[str, str, Callable[[], Any]]] = [
        ("unsafe_helper_bridge", "mlx_to_tinygrad", lambda s=src_mx: tinygrad_from_mlx_fast(s, tg_dtype)),
        ("single_entry_bridge", "mlx_to_tinygrad", lambda s=src_mx: tinygrad_from_mlx_single_entry(s, tg_dtype)),
        ("fresh_wrapper_then_use_sum", "mlx_to_tinygrad", lambda s=src_mx: tinygrad_consume_sum(tinygrad_from_mlx_fast(s, tg_dtype))),
        ("rebindable_slot_bridge", "mlx_to_tinygrad", lambda alt=mx_slot_sources, b=mx_slot_borrower: tinygrad_from_mlx_reuse(alt.next(), b)),
        ("rebindable_slot_then_use_sum", "mlx_to_tinygrad",
         lambda alt=mx_slot_sources, b=mx_slot_borrower: tinygrad_consume_sum(tinygrad_from_mlx_reuse(alt.next(), b))),
        ("borrower_ring4_bridge", "mlx_to_tinygrad",
         lambda alt=mx_slot_sources, ring=mx_ring: tinygrad_from_mlx_reuse(alt.next(), ring.next())),
        ("borrower_ring4_then_use_sum", "mlx_to_tinygrad",
         lambda alt=mx_slot_sources, ring=mx_ring: tinygrad_consume_sum(tinygrad_from_mlx_reuse(alt.next(), ring.next()))),
        ("lease_pool_acquire_release", "mlx_to_tinygrad",
         lambda alt=mx_slot_sources, pool=lease_pool: tinygrad_from_mlx_lease_acquire_release(alt.next(), pool)),
        ("lease_pool_then_use_sum", "mlx_to_tinygrad",
         lambda alt=mx_slot_sources, pool=lease_pool: tinygrad_from_mlx_lease_then_use_sum(alt.next(), pool)),
        ("scoped_handoff_noop", "mlx_to_tinygrad",
         lambda alt=mx_slot_sources, pools=scoped_handoff_pools, dtype=tg_dtype: tinygrad_from_mlx_scoped_handoff_noop(alt.next(), pools, dtype)),
        ("scoped_handoff_then_use_sum", "mlx_to_tinygrad",
         lambda alt=mx_slot_sources, pools=scoped_handoff_pools, dtype=tg_dtype: tinygrad_from_mlx_scoped_handoff_then_use_sum(alt.next(), pools, dtype)),
        ("copy_pool_acquire_release", "mlx_to_tinygrad",
         lambda alt=mx_slot_sources, pool=copy_lease_pool: tinygrad_from_mlx_copy_pool_acquire_release(alt.next(), pool)),
        ("copy_pool_then_use_sum", "mlx_to_tinygrad",
         lambda alt=mx_slot_sources, pool=copy_lease_pool: tinygrad_from_mlx_copy_pool_then_use_sum(alt.next(), pool)),
        ("scoped_copy_handoff_noop", "mlx_to_tinygrad",
         lambda alt=mx_slot_sources, pools=scoped_copy_handoff_pools, dtype=tg_dtype: tinygrad_from_mlx_scoped_copy_handoff_noop(alt.next(), pools, dtype)),
        ("scoped_copy_handoff_then_use_sum", "mlx_to_tinygrad",
         lambda alt=mx_slot_sources, pools=scoped_copy_handoff_pools, dtype=tg_dtype: tinygrad_from_mlx_scoped_copy_handoff_then_use_sum(alt.next(), pools, dtype)),
        ("unsafe_helper_legacy", "mlx_to_tinygrad", lambda s=src_mx: tinygrad_from_mlx_legacy(s, tg_dtype)),
        ("memoryview_copy", "mlx_to_tinygrad", lambda s=src_mx: tinygrad_from_mlx_copy(s, tg_dtype)),
        ("numpy_baseline", "mlx_to_tinygrad", lambda s=src_mx: tinygrad_from_mlx_numpy(s)),
        ("unsafe_helper_bridge", "tinygrad_to_mlx", lambda s=src_tg: mlx_from_tinygrad_alias_only(s, mx_dtype)),
        ("unsafe_helper_maybe_copy", "tinygrad_to_mlx", lambda s=src_tg: mlx_from_tinygrad_maybe_copy(s, mx_dtype)),
        ("memoryview_copy", "tinygrad_to_mlx", lambda s=src_tg: mlx_from_tinygrad_copy(s)),
        ("numpy_baseline", "tinygrad_to_mlx", lambda s=src_tg: mlx_from_tinygrad_numpy(s)),
        ("export_helper_only", "mlx_to_tinygrad", lambda s=src_mx: mlx_export_storage(s)),
        ("import_helper_fast_only", "mlx_to_tinygrad", lambda st=mx_storage, s=src_mx: tinygrad_import_from_storage_fast(st, tg_dtype, s)),
        ("rebindable_slot_import_only", "mlx_to_tinygrad",
         lambda alt=mx_slot_storage, b=mx_slot_borrower: tinygrad_import_from_storage_reuse(*alt.next(), borrower=b)),
        ("borrower_ring4_import_only", "mlx_to_tinygrad",
         lambda alt=mx_slot_storage, ring=mx_ring: tinygrad_import_from_storage_reuse(*alt.next(), borrower=ring.next())),
        ("import_helper_legacy_only", "mlx_to_tinygrad", lambda st=mx_storage, s=src_mx: tinygrad_import_from_storage(st, tg_dtype, s)),
        ("export_helper_only", "tinygrad_to_mlx", lambda s=src_tg: tinygrad_export_storage(s)),
        ("import_helper_only", "tinygrad_to_mlx", lambda st=tg_storage, s=src_tg: mlx_import_from_storage_alias_only(st, mx_dtype, s)),
        ("import_helper_maybe_copy_only", "tinygrad_to_mlx", lambda st=tg_storage, s=src_tg: mlx_import_from_storage(st, mx_dtype, s)),
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
          f"{stats['avg_us']:.3f},{stats['stddev_us']:.3f},{int(stats['iters'])}"
        )
  finally:
    gc.enable()


if __name__ == "__main__":
  main()
