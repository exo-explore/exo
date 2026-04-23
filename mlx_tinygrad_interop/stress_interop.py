from __future__ import annotations

import argparse
import gc
import tracemalloc
from typing import Any, cast

import mlx.core as mx
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer

from mlx_tinygrad_interop.lease_pool import MlxToTinygradLeasePools

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
    np.testing.assert_allclose(actual, expected, rtol=5e-3 if expected.dtype == np.float16 else 1e-5, atol=5e-3 if expected.dtype == np.float16 else 1e-6, err_msg=name)
  else:
    np.testing.assert_array_equal(actual, expected, err_msg=name)


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
    return np_dtype.type(rng.uniform(-2.0, 2.0))
  if np.issubdtype(np_dtype, np.unsignedinteger):
    return np_dtype.type(int(rng.integers(0, 4)))
  return np_dtype.type(int(rng.integers(-3, 4)))


def random_ops(rng: np.random.Generator, shape: tuple[int, ...], np_dtype: np.dtype[Any]) -> list[tuple[str, Any]]:
  ops: list[tuple[str, Any]] = []
  cur_shape = shape
  if len(cur_shape) > 1 and rng.random() < 0.7:
    perm = tuple(int(x) for x in rng.permutation(len(cur_shape)))
    ops.append(("transpose", perm))
    cur_shape = tuple(cur_shape[i] for i in perm)
  if len(cur_shape) > 1 and rng.random() < 0.7:
    ops.append(("reshape", tuple(reversed(cur_shape))))
    cur_shape = tuple(reversed(cur_shape))
  ops.append(("add", random_scalar(rng, np_dtype)))
  ops.append(("mul", random_scalar(rng, np_dtype)))
  if rng.random() < 0.5:
    ops.append(("add", random_scalar(rng, np_dtype)))
  return ops


def apply_numpy_ops(x: np.ndarray, ops: list[tuple[str, Any]]) -> np.ndarray:
  out = x.copy()
  for op, arg in ops:
    if op == "transpose": out = np.transpose(out, arg)
    elif op == "reshape": out = out.reshape(arg)
    elif op == "add": out = out + arg
    elif op == "mul": out = out * arg
    else: raise RuntimeError(f"unknown op {op}")
  return out


def apply_tinygrad_ops(x: Tensor, ops: list[tuple[str, Any]]) -> Tensor:
  out = x
  for op, arg in ops:
    if op == "transpose": out = out.permute(arg)
    elif op == "reshape": out = out.reshape(arg)
    elif op == "add": out = out + arg
    elif op == "mul": out = out * arg
    else: raise RuntimeError(f"unknown op {op}")
  return out.realize()


def apply_mlx_ops(x: Any, ops: list[tuple[str, Any]]) -> Any:
  out = x
  for op, arg in ops:
    if op == "transpose": out = mx.transpose(out, arg)
    elif op == "reshape": out = mx.reshape(out, arg)
    elif op == "add": out = out + arg
    elif op == "mul": out = out * arg
    else: raise RuntimeError(f"unknown op {op}")
  mx.eval(out)
  return out


def run_case(case_index: int, rng: np.random.Generator, mx_dtype: Any, tg_dtype: Any, np_dtype: np.dtype[Any],
             max_elements: int, pools: MlxToTinygradLeasePools) -> None:
  shape = random_shape(rng, max_elements)
  values = random_values(rng, np_dtype, shape)
  ops = random_ops(rng, shape, np_dtype)

  mx_source = make_mlx_source(values, mx_dtype, rng)
  tg_source = Tensor(values, device="METAL", dtype=tg_dtype).realize()
  Device["METAL"].synchronize()

  expected_tg_raw = Tensor(np.array(mx_source, copy=True), device="METAL", dtype=tg_dtype).realize()
  expected_mx_raw = mx.array(tg_source.numpy())
  expected_after_ops = apply_numpy_ops(values.astype(np.float32), ops)

  tg_fast = tinygrad_from_mlx_fast(mx_source, tg_dtype)
  tg_single = tinygrad_from_mlx_single_entry(mx_source, tg_dtype)
  with pools.acquire_from_mlx(mx_source, tg_dtype=tg_dtype) as lease:
    tg_lease = lease.tensor
    assert_array_close(f"case {case_index} mlx->tinygrad lease raw", tg_lease.numpy(), values)
    assert_array_close(
      f"case {case_index} mlx->tinygrad lease ops",
      apply_tinygrad_ops(tg_lease.cast(dtypes.float32), ops).numpy(),
      expected_after_ops,
    )

  assert_array_close(f"case {case_index} mlx->tinygrad fast raw", tg_fast.numpy(), expected_tg_raw.numpy())
  assert_array_close(
    f"case {case_index} mlx->tinygrad fast ops",
    apply_tinygrad_ops(tg_fast.cast(dtypes.float32), ops).numpy(),
    expected_after_ops,
  )
  assert_array_close(f"case {case_index} mlx->tinygrad single raw", tg_single.numpy(), expected_tg_raw.numpy())
  assert_array_close(
    f"case {case_index} mlx->tinygrad single ops",
    apply_tinygrad_ops(tg_single.cast(dtypes.float32), ops).numpy(),
    expected_after_ops,
  )

  mx_alias = mlx_from_tinygrad_alias_only(tg_source, mx_dtype)
  mx_maybe_copy = mlx_from_tinygrad_maybe_copy(tg_source, mx_dtype)
  mx_copy = mlx_from_tinygrad_copy(tg_source)
  assert_array_close(f"case {case_index} tinygrad->mlx alias raw", np.array(mx_alias), np.array(expected_mx_raw))
  assert_array_close(
    f"case {case_index} tinygrad->mlx alias ops",
    np.array(apply_mlx_ops(mx.astype(mx_alias, mx.float32), ops)),
    expected_after_ops,
  )
  assert_array_close(f"case {case_index} tinygrad->mlx maybe_copy raw", np.array(mx_maybe_copy), np.array(expected_mx_raw))
  assert_array_close(
    f"case {case_index} tinygrad->mlx maybe_copy ops",
    np.array(apply_mlx_ops(mx.astype(mx_maybe_copy, mx.float32), ops)),
    expected_after_ops,
  )
  assert_array_close(f"case {case_index} tinygrad->mlx copy raw", np.array(mx_copy), np.array(expected_mx_raw))
  assert_array_close(
    f"case {case_index} tinygrad->mlx copy ops",
    np.array(apply_mlx_ops(mx.astype(mx_copy, mx.float32), ops)),
    expected_after_ops,
  )


def run_soak(rng: np.random.Generator, dtype_names: list[str], iterations: int, max_elements: int) -> None:
  pools = MlxToTinygradLeasePools(capacity_per_key=8, synchronize_on_release=False)
  checksum = 0.0
  for iteration in range(iterations):
    dtype_name = dtype_names[iteration % len(dtype_names)]
    mx_dtype, tg_dtype, np_dtype = DTYPES[dtype_name]
    shape = random_shape(rng, max_elements)
    values = random_values(rng, np_dtype, shape)
    mx_source = make_mlx_source(values, mx_dtype, rng)
    tg_source = Tensor(values, device="METAL", dtype=tg_dtype).realize()
    Device["METAL"].synchronize()

    with pools.acquire_from_mlx(mx_source, tg_dtype=tg_dtype) as lease:
      checksum += float(apply_tinygrad_ops(lease.tensor.cast(dtypes.float32), [("add", np.float32(1))]).sum().item())
    checksum += float(np.array(mlx_from_tinygrad_copy(tg_source)).astype(np.float64).sum())
    if (iteration + 1) % 64 == 0:
      gc.collect()
  print(f"# soak_checksum={checksum:.6f} pool_count={pools.pool_count}")


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
  pools = MlxToTinygradLeasePools(capacity_per_key=8, synchronize_on_release=True)

  for case_index in range(args.cases):
    dtype_name = dtype_names[case_index % len(dtype_names)]
    run_case(case_index, rng, *DTYPES[dtype_name], max_elements=args.max_elements, pools=pools)

  run_soak(rng, dtype_names, iterations=args.soak_iterations, max_elements=args.max_elements)
  gc.collect()
  after_cur, after_peak = tracemalloc.get_traced_memory()
  print(
    "# stress_ok"
    f" cases={args.cases}"
    f" soak_iterations={args.soak_iterations}"
    f" tracemalloc_current={after_cur - before_cur}"
    f" tracemalloc_peak={after_peak - before_peak}"
  )


if __name__ == "__main__":
  main()
