from __future__ import annotations

import argparse
import gc
import resource
import sys
import tracemalloc
from typing import Any, cast

import mlx.core as mx
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer

try:
  from mlx_tinygrad_interop.lease_pool import MlxToTinygradCopyLeasePools, MlxToTinygradLeasePools
except ModuleNotFoundError:
  from lease_pool import MlxToTinygradCopyLeasePools, MlxToTinygradLeasePools

DTYPES: dict[str, tuple[Any, Any, np.dtype[Any]]] = {
  "float16": (mx.float16, dtypes.float16, np.dtype(np.float16)),
  "float32": (mx.float32, dtypes.float32, np.dtype(np.float32)),
  "int32": (mx.int32, dtypes.int32, np.dtype(np.int32)),
  "uint8": (mx.uint8, dtypes.uint8, np.dtype(np.uint8)),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Randomized correctness and soak checks for MLX <-> tinygrad interop.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--cases", type=int, default=64, help="Number of randomized correctness cases.")
  parser.add_argument("--soak-iterations", type=int, default=512, help="Number of repeated pool/copy iterations after correctness checks.")
  parser.add_argument("--max-elements", type=int, default=4096, help="Upper bound on random tensor element count.")
  parser.add_argument("--max-pools", type=int, default=16, help="Maximum keyed pools to retain for alias and copy handoff managers.")
  parser.add_argument("--dtypes", default="float16,float32,int32,uint8", help="Comma-separated dtype names to exercise.")
  return parser.parse_args()


def tinygrad_zero_copy_memoryview(t: Tensor) -> memoryview:
  buf = cast(Buffer, t.uop.buffer).ensure_allocated()
  assert t.dtype.base.fmt is not None, f"no buffer format for dtype {t.dtype.base}"
  return buf.as_memoryview(force_zero_copy=True).cast(t.dtype.base.fmt, t.shape)


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


def mlx_from_tinygrad_alias_only(t: Tensor, mx_dtype: Any) -> Any:
  storage = t._unsafe_metal_storage()
  return mx.metal._unsafe_array_from_ptr_alias_only(
    int(storage["raw_ptr"]),
    tuple(storage["shape"]),
    mx_dtype,
    owner=t,
  )


def mlx_from_tinygrad_maybe_copy(t: Tensor, mx_dtype: Any) -> Any:
  storage = t._unsafe_metal_storage()
  return mx.metal._unsafe_array_from_ptr(
    int(storage["raw_ptr"]),
    tuple(storage["shape"]),
    mx_dtype,
    owner=t,
  )


def mlx_from_tinygrad_copy(t: Tensor) -> Any:
  return mx.array(tinygrad_zero_copy_memoryview(t))


def assert_array_close(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
  if np.issubdtype(expected.dtype, np.floating):
    if expected.dtype == np.float16:
      rtol, atol = 5e-3, 5e-3
    else:
      # Mixed matmul/reduction chains across NumPy/MLX/tinygrad can drift by a
      # few ulps from accumulation-order differences even when the conversion is
      # correct. Keep float32 strict, but not unrealistically bit-exact.
      rtol, atol = 5e-5, 1e-5
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
  else:
    np.testing.assert_array_equal(actual, expected, err_msg=name)


def rss_max_bytes() -> int:
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return int(rss if sys.platform == "darwin" else rss * 1024)


def mlx_memory_snapshot() -> dict[str, int]:
  stats: dict[str, int] = {}
  for name in ("get_active_memory", "get_cache_memory", "get_peak_memory"):
    fn = getattr(mx, name, None)
    if callable(fn): stats[name] = int(fn())
  return stats


def random_shape(rng: np.random.Generator, max_elements: int) -> tuple[int, ...]:
  ndim = int(rng.integers(1, 5))
  remaining = max(1, int(rng.integers(1, max_elements + 1)))
  shape: list[int] = []
  for dim_index in range(ndim):
    dims_left = ndim - dim_index
    if dims_left == 1:
      shape.append(remaining)
      break
    dim = int(rng.integers(1, max(2, int(round(remaining ** (1 / dims_left))) + 2)))
    shape.append(dim)
    remaining = max(1, remaining // dim)
  return tuple(shape)


def random_values(rng: np.random.Generator, np_dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
  if np.issubdtype(np_dtype, np.floating):
    data = rng.standard_normal(np.prod(shape, dtype=np.int64)).astype(np.float32)
    return data.astype(np_dtype).reshape(shape)
  if np.issubdtype(np_dtype, np.unsignedinteger):
    info = np.iinfo(np_dtype)
    return rng.integers(0, min(info.max, 255) + 1, size=shape, dtype=np_dtype)
  info = np.iinfo(np_dtype)
  return rng.integers(max(info.min, -128), min(info.max, 127) + 1, size=shape, dtype=np_dtype)


def make_mlx_source(values: np.ndarray, mx_dtype: Any, rng: np.random.Generator) -> Any:
  flat = values.reshape(-1)
  if flat.size > 0 and rng.random() < 0.5:
    offset_elems = int(rng.integers(1, 33))
    suffix = int(rng.integers(1, 33))
    backing = np.zeros(offset_elems + flat.size + suffix, dtype=flat.dtype)
    backing[offset_elems:offset_elems + flat.size] = flat
    base = mx.array(backing, dtype=mx_dtype)
    view = base[offset_elems:offset_elems + flat.size].reshape(values.shape)
    mx.eval(view)
    return view
  return mx.array(values, dtype=mx_dtype)


def random_scalar(rng: np.random.Generator, np_dtype: np.dtype[Any]) -> Any:
  if np.issubdtype(np_dtype, np.floating):
    return np_dtype.type(rng.uniform(-2.0, 2.0)).item()
  if np.issubdtype(np_dtype, np.unsignedinteger):
    return int(rng.integers(0, 4))
  return int(rng.integers(-3, 4))


def random_slice(axis_size: int, rng: np.random.Generator) -> tuple[int, int]:
  start = int(rng.integers(0, axis_size))
  end = int(rng.integers(start + 1, axis_size + 1))
  return start, end


def random_ops(rng: np.random.Generator, shape: tuple[int, ...], np_dtype: np.dtype[Any]) -> list[tuple[str, Any]]:
  ops: list[tuple[str, Any]] = []
  cur_shape = shape
  if len(cur_shape) > 1 and rng.random() < 0.7:
    perm = tuple(int(x) for x in rng.permutation(len(cur_shape)))
    ops.append(("transpose", perm))
    cur_shape = tuple(cur_shape[i] for i in perm)
  if len(cur_shape) > 1 and rng.random() < 0.6:
    reshaped = tuple(reversed(cur_shape))
    ops.append(("reshape", reshaped))
    cur_shape = reshaped
  if any(dim > 1 for dim in cur_shape) and rng.random() < 0.5:
    axis = int(rng.choice([i for i, dim in enumerate(cur_shape) if dim > 1]))
    start, end = random_slice(cur_shape[axis], rng)
    ops.append(("slice", (axis, start, end)))
    cur_shape = cur_shape[:axis] + (end - start,) + cur_shape[axis + 1:]
  ops.append(("add", random_scalar(rng, np_dtype)))
  ops.append(("mul", random_scalar(rng, np_dtype)))
  if cur_shape and rng.random() < 0.7:
    bshape = tuple(dim if rng.random() < 0.5 else 1 for dim in cur_shape)
    ops.append(("broadcast_add", random_values(rng, np_dtype, bshape)))
  if rng.random() < 0.5:
    ops.append(("relu", None))
  if np.issubdtype(np_dtype, np.floating) and cur_shape and rng.random() < 0.35:
    out_cols = int(rng.integers(1, min(8, cur_shape[-1]) + 1))
    weight = random_values(rng, np_dtype, (cur_shape[-1], out_cols))
    ops.append(("matmul_lastdim", weight))
    cur_shape = (int(np.prod(cur_shape[:-1], dtype=np.int64)), out_cols)
  if cur_shape and rng.random() < 0.5:
    axis = int(rng.integers(0, len(cur_shape)))
    keepdim = bool(rng.integers(0, 2))
    ops.append(("sum", (axis, keepdim)))
    cur_shape = cur_shape[:axis] + ((1,) if keepdim else ()) + cur_shape[axis + 1:]
  if cur_shape and rng.random() < 0.35:
    axis = int(rng.integers(0, len(cur_shape)))
    ops.append(("concat_self", axis))
  return ops


def _slice_spec(shape: tuple[int, ...], axis: int, start: int, end: int) -> tuple[slice, ...]:
  return tuple(slice(start, end) if i == axis else slice(None) for i in range(len(shape)))


def apply_numpy_ops(x: np.ndarray, ops: list[tuple[str, Any]]) -> np.ndarray:
  out = x.copy()
  for op, arg in ops:
    if op == "transpose":
      out = np.transpose(out, arg)
    elif op == "reshape":
      out = out.reshape(arg)
    elif op == "slice":
      axis, start, end = arg
      out = out[_slice_spec(out.shape, axis, start, end)]
    elif op == "add":
      out = out + arg
    elif op == "mul":
      out = out * arg
    elif op == "broadcast_add":
      out = out + arg
    elif op == "relu":
      out = np.maximum(out, 0)
    elif op == "matmul_lastdim":
      # NumPy's `@` path on the current macOS validation host produced an
      # incorrect all-zero result for a valid contiguous float32 case that MLX,
      # tinygrad, and `np.einsum` all agreed on. Use einsum here so the stress
      # harness keeps a trustworthy numerical baseline.
      out = np.einsum("ik,kj->ij", out.reshape(-1, out.shape[-1]), arg, optimize=True)
    elif op == "sum":
      axis, keepdim = arg
      out = out.sum(axis=axis, keepdims=keepdim)
    elif op == "concat_self":
      out = np.concatenate([out, out], axis=arg)
    else:
      raise RuntimeError(f"unknown op {op}")
  return out


def apply_tinygrad_ops(x: Tensor, ops: list[tuple[str, Any]]) -> Tensor:
  out = x
  for op, arg in ops:
    if op == "transpose":
      out = out.permute(arg)
    elif op == "reshape":
      out = out.reshape(arg)
    elif op == "slice":
      axis, start, end = arg
      out = out[_slice_spec(out.shape, axis, start, end)]
    elif op == "add":
      out = out + arg
    elif op == "mul":
      out = out * arg
    elif op == "broadcast_add":
      out = out + Tensor(arg, device=out.device, dtype=out.dtype)
    elif op == "relu":
      out = out.relu()
    elif op == "matmul_lastdim":
      out = out.reshape(-1, out.shape[-1]) @ Tensor(arg, device=out.device, dtype=out.dtype)
    elif op == "sum":
      axis, keepdim = arg
      out = out.sum(axis=axis, keepdim=keepdim)
    elif op == "concat_self":
      out = out.cat(out, dim=arg)
    else:
      raise RuntimeError(f"unknown op {op}")
  return out.realize()


def apply_mlx_ops(x: Any, ops: list[tuple[str, Any]]) -> Any:
  out = x
  for op, arg in ops:
    if op == "transpose":
      out = mx.transpose(out, arg)
    elif op == "reshape":
      out = mx.reshape(out, arg)
    elif op == "slice":
      axis, start, end = arg
      out = out[_slice_spec(out.shape, axis, start, end)]
    elif op == "add":
      out = out + arg
    elif op == "mul":
      out = out * arg
    elif op == "broadcast_add":
      out = out + mx.array(arg, dtype=out.dtype)
    elif op == "relu":
      out = mx.maximum(out, 0)
    elif op == "matmul_lastdim":
      out = mx.reshape(out, (-1, out.shape[-1])) @ mx.array(arg, dtype=out.dtype)
    elif op == "sum":
      axis, keepdim = arg
      out = mx.sum(out, axis=axis, keepdims=keepdim)
    elif op == "concat_self":
      out = mx.concatenate([out, out], axis=arg)
    else:
      raise RuntimeError(f"unknown op {op}")
  mx.eval(out)
  return out


def run_case(case_index: int, rng: np.random.Generator, mx_dtype: Any, tg_dtype: Any, np_dtype: np.dtype[Any],
             max_elements: int, alias_pools: MlxToTinygradLeasePools, copy_pools: MlxToTinygradCopyLeasePools) -> None:
  shape = random_shape(rng, max_elements)
  values = random_values(rng, np_dtype, shape)
  ops = random_ops(rng, shape, np_dtype)

  mx_source = make_mlx_source(values, mx_dtype, rng)
  tg_source = Tensor(values, device="METAL", dtype=tg_dtype).realize()
  Device["METAL"].synchronize()

  expected_raw = values
  expected_after_ops = apply_numpy_ops(values, ops)
  tg_fast = tinygrad_from_mlx_fast(mx_source, tg_dtype)
  tg_single = tinygrad_from_mlx_single_entry(mx_source, tg_dtype)
  assert_array_close(f"case {case_index} mlx->tinygrad fast raw", tg_fast.numpy(), expected_raw)
  assert_array_close(f"case {case_index} mlx->tinygrad single raw", tg_single.numpy(), expected_raw)
  assert_array_close(
    f"case {case_index} mlx->tinygrad fast ops",
    apply_tinygrad_ops(tg_fast, ops).numpy(),
    expected_after_ops,
  )
  assert_array_close(
    f"case {case_index} mlx->tinygrad single ops",
    apply_tinygrad_ops(tg_single, ops).numpy(),
    expected_after_ops,
  )

  alias_result = alias_pools.run_with_mlx_tensor(
    mx_source,
    tg_dtype=tg_dtype,
    fn=lambda tg: apply_tinygrad_ops(tg, ops),
  )
  copy_result = copy_pools.run_with_mlx_tensor(
    mx_source,
    tg_dtype=tg_dtype,
    fn=lambda tg: apply_tinygrad_ops(tg, ops),
  )
  assert_array_close(f"case {case_index} alias scoped ops", alias_result.numpy(), expected_after_ops)
  assert_array_close(f"case {case_index} copy scoped ops", copy_result.numpy(), expected_after_ops)

  alias_roundtrip = np.array(mlx_from_tinygrad_copy(alias_result.cast(tg_dtype).realize()))
  copy_roundtrip = np.array(mlx_from_tinygrad_copy(copy_result.cast(tg_dtype).realize()))
  assert_array_close(f"case {case_index} alias roundtrip raw", alias_roundtrip, np.array(alias_result.cast(tg_dtype).numpy(), copy=True))
  assert_array_close(f"case {case_index} copy roundtrip raw", copy_roundtrip, np.array(copy_result.cast(tg_dtype).numpy(), copy=True))

  mx_alias = mlx_from_tinygrad_alias_only(tg_source, mx_dtype)
  mx_maybe_copy = mlx_from_tinygrad_maybe_copy(tg_source, mx_dtype)
  mx_copy = mlx_from_tinygrad_copy(tg_source)
  assert_array_close(f"case {case_index} tinygrad->mlx alias raw", np.array(mx_alias), expected_raw)
  assert_array_close(f"case {case_index} tinygrad->mlx maybe_copy raw", np.array(mx_maybe_copy), expected_raw)
  assert_array_close(f"case {case_index} tinygrad->mlx copy raw", np.array(mx_copy), expected_raw)
  assert_array_close(
    f"case {case_index} tinygrad->mlx alias ops",
    np.array(apply_mlx_ops(mx_alias, ops)),
    expected_after_ops,
  )
  assert_array_close(
    f"case {case_index} tinygrad->mlx maybe_copy ops",
    np.array(apply_mlx_ops(mx_maybe_copy, ops)),
    expected_after_ops,
  )
  assert_array_close(
    f"case {case_index} tinygrad->mlx copy ops",
    np.array(apply_mlx_ops(mx_copy, ops)),
    expected_after_ops,
  )


def run_soak(rng: np.random.Generator, dtype_names: list[str], iterations: int, max_elements: int, max_pools: int) -> tuple[float, list[tuple[int, dict[str, int], int, int, int]]]:
  alias_pools = MlxToTinygradLeasePools(capacity_per_key=8, max_pools=max_pools, synchronize_on_release=True)
  copy_pools = MlxToTinygradCopyLeasePools(capacity_per_key=8, max_pools=max_pools, synchronize_on_release=True)
  checksum = 0.0
  native_checkpoints: list[tuple[int, dict[str, int], int, int, int]] = []
  for iteration in range(iterations):
    dtype_name = dtype_names[iteration % len(dtype_names)]
    mx_dtype, tg_dtype, np_dtype = DTYPES[dtype_name]
    shape = random_shape(rng, max_elements)
    values = random_values(rng, np_dtype, shape)
    mx_source = make_mlx_source(values, mx_dtype, rng)
    tg_source = Tensor(values, device="METAL", dtype=tg_dtype).realize()
    Device["METAL"].synchronize()

    checksum += float(alias_pools.run_with_mlx_tensor(
      mx_source,
      tg_dtype=tg_dtype,
      fn=lambda tg: apply_tinygrad_ops(tg, [("add", 1.0 if np.issubdtype(np_dtype, np.floating) else 1)]).sum(),
    ).item())
    checksum += float(copy_pools.run_with_mlx_tensor(
      mx_source,
      tg_dtype=tg_dtype,
      fn=lambda tg: apply_tinygrad_ops(tg, [("mul", 1.0 if np.issubdtype(np_dtype, np.floating) else 1)]).sum(),
    ).item())
    checksum += float(np.array(mlx_from_tinygrad_copy(tg_source)).astype(np.float64).sum())
    if (iteration + 1) % 64 == 0:
      gc.collect()
      native_checkpoints.append((iteration + 1, mlx_memory_snapshot(), rss_max_bytes(), alias_pools.pool_count, copy_pools.pool_count))

  if alias_pools.pool_count > max_pools:
    raise AssertionError(f"alias pool registry exceeded cap: {alias_pools.pool_count} > {max_pools}")
  if copy_pools.pool_count > max_pools:
    raise AssertionError(f"copy pool registry exceeded cap: {copy_pools.pool_count} > {max_pools}")
  print(f"# soak_checksum={checksum:.6f} alias_pool_count={alias_pools.pool_count} copy_pool_count={copy_pools.pool_count}")
  return checksum, native_checkpoints


def main() -> None:
  args = parse_args()
  if not mx.metal.is_available():
    raise RuntimeError("Metal is not available")

  dtype_names = [name.strip() for name in args.dtypes.split(",") if name.strip()]
  missing = [name for name in dtype_names if name not in DTYPES]
  if missing:
    raise RuntimeError(f"unknown dtype(s): {', '.join(missing)}")

  rng = np.random.default_rng(args.seed)
  tracemalloc.start()
  before_cur, before_peak = tracemalloc.get_traced_memory()
  reset_peak = getattr(mx, "reset_peak_memory", None)
  if callable(reset_peak): reset_peak()
  native_before = mlx_memory_snapshot()
  rss_before = rss_max_bytes()
  alias_pools = MlxToTinygradLeasePools(capacity_per_key=8, max_pools=args.max_pools, synchronize_on_release=True)
  copy_pools = MlxToTinygradCopyLeasePools(capacity_per_key=8, max_pools=args.max_pools, synchronize_on_release=True)

  for case_index in range(args.cases):
    dtype_name = dtype_names[case_index % len(dtype_names)]
    run_case(case_index, rng, *DTYPES[dtype_name], max_elements=args.max_elements, alias_pools=alias_pools, copy_pools=copy_pools)

  _, native_checkpoints = run_soak(rng, dtype_names, iterations=args.soak_iterations, max_elements=args.max_elements, max_pools=args.max_pools)
  gc.collect()
  after_cur, after_peak = tracemalloc.get_traced_memory()
  native_after = mlx_memory_snapshot()
  rss_after = rss_max_bytes()
  active_peak = max([native_before.get("get_active_memory", 0), native_after.get("get_active_memory", 0),
                     *[stats.get("get_active_memory", 0) for _, stats, _, _, _ in native_checkpoints]], default=0)
  cache_peak = max([native_before.get("get_cache_memory", 0), native_after.get("get_cache_memory", 0),
                    *[stats.get("get_cache_memory", 0) for _, stats, _, _, _ in native_checkpoints]], default=0)
  rss_peak = max([rss_before, rss_after, *[rss for _, _, rss, _, _ in native_checkpoints]], default=0)
  alias_pool_peak = max([alias_pools.pool_count, *[alias_pool_count for _, _, _, alias_pool_count, _ in native_checkpoints]], default=0)
  copy_pool_peak = max([copy_pools.pool_count, *[copy_pool_count for _, _, _, _, copy_pool_count in native_checkpoints]], default=0)
  print(
    "# native_memory"
    f" mlx_active_start={native_before.get('get_active_memory', -1)}"
    f" mlx_active_end={native_after.get('get_active_memory', -1)}"
    f" mlx_active_peak={active_peak}"
    f" mlx_cache_start={native_before.get('get_cache_memory', -1)}"
    f" mlx_cache_end={native_after.get('get_cache_memory', -1)}"
    f" mlx_cache_peak={cache_peak}"
    f" mlx_reported_peak={native_after.get('get_peak_memory', -1)}"
    f" rss_max_start={rss_before}"
    f" rss_max_end={rss_after}"
    f" rss_max_peak={rss_peak}"
    f" alias_pool_peak={alias_pool_peak}"
    f" copy_pool_peak={copy_pool_peak}"
  )
  print(
    "# stress_ok"
    f" cases={args.cases}"
    f" soak_iterations={args.soak_iterations}"
    f" max_pools={args.max_pools}"
    f" tracemalloc_current={after_cur - before_cur}"
    f" tracemalloc_peak={after_peak - before_peak}"
  )


if __name__ == "__main__":
  main()
