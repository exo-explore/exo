import unittest

import mlx.core as mx
import numpy as np
from tinygrad import Tensor, dtypes

from mlx_tinygrad_interop.lease_pool import MlxToTinygradLeaseKey, MlxToTinygradLeasePool, MlxToTinygradLeasePools


@unittest.skipUnless(mx.metal.is_available(), "Metal is not available")
class TestMlxTinygradLeasePool(unittest.TestCase):
  def test_checked_rebind_requires_shape_and_dtype(self):
    array = mx.array(np.arange(16, dtype=np.float32), dtype=mx.float32)
    storage = mx.metal._unsafe_export_storage(array)
    borrower = Tensor._unsafe_metal_borrower(
      int(storage["mtl_buffer_ptr"]),
      tuple(storage["shape"]),
      dtype=dtypes.float32,
      byte_offset=int(storage["offset_bytes"]),
      buffer_nbytes=int(storage["buffer_nbytes"]),
      owner=array,
    )

    with self.assertRaises(TypeError):
      borrower.rebind(
        int(storage["mtl_buffer_ptr"]),
        owner=array,
        byte_offset=int(storage["offset_bytes"]),
        buffer_nbytes=int(storage["buffer_nbytes"]),
      )

  def test_lease_pool_evaluates_lazy_array_on_acquire(self):
    lazy = mx.arange(32, dtype=mx.float32) + mx.float32(3)
    pool = MlxToTinygradLeasePool.from_mlx(lazy, tg_dtype=dtypes.float32, capacity=1)

    with pool.acquire_from_mlx(lazy) as lease:
      np.testing.assert_array_equal(lease.tensor.numpy(), np.arange(32, dtype=np.float32) + np.float32(3))

  def test_lease_pool_capacity_requires_release(self):
    first = mx.array(np.arange(16, dtype=np.float32), dtype=mx.float32)
    second = mx.array(np.arange(16, dtype=np.float32) + np.float32(1), dtype=mx.float32)
    pool = MlxToTinygradLeasePool.from_mlx(first, tg_dtype=dtypes.float32, capacity=1)

    lease = pool.acquire_from_mlx(first)
    with self.assertRaisesRegex(RuntimeError, "still in use"):
      pool.acquire_from_mlx(second)
    lease.release(synchronize=False)

    with pool.acquire_from_mlx(second) as second_lease:
      np.testing.assert_array_equal(second_lease.tensor.numpy(), np.arange(16, dtype=np.float32) + np.float32(1))

  def test_lease_release_invalidates_future_access(self):
    array = mx.array(np.arange(8, dtype=np.float32), dtype=mx.float32)
    pool = MlxToTinygradLeasePool.from_mlx(array, tg_dtype=dtypes.float32, capacity=1)

    lease = pool.acquire_from_mlx(array)
    self.assertEqual(lease.generation, 1)
    lease.release(synchronize=False)

    with self.assertRaisesRegex(RuntimeError, "already released"):
      _ = lease.tensor
    with self.assertRaisesRegex(RuntimeError, "already released"):
      lease.release(synchronize=False)

  def test_keyed_lease_pools_bucket_by_contract(self):
    flat = mx.array(np.arange(16, dtype=np.float32), dtype=mx.float32)
    matrix = mx.array(np.arange(16, dtype=np.float32).reshape(4, 4), dtype=mx.float32)
    pools = MlxToTinygradLeasePools(capacity_per_key=2, synchronize_on_release=False)

    with pools.acquire_from_mlx(flat, tg_dtype=dtypes.float32) as flat_lease:
      self.assertEqual(flat_lease.key, MlxToTinygradLeaseKey((16,), "float32", dtypes.float32.base.name, 0))

    with pools.acquire_from_mlx(matrix, tg_dtype=dtypes.float32) as matrix_lease:
      self.assertEqual(matrix_lease.key, MlxToTinygradLeaseKey((4, 4), "float32", dtypes.float32.base.name, 0))

    self.assertEqual(pools.pool_count, 2)


if __name__ == "__main__":
  unittest.main()
