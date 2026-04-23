import unittest

import mlx.core as mx
import numpy as np
from tinygrad import Tensor, dtypes

from mlx_tinygrad_interop.lease_pool import (
  MlxToTinygradCopyKey,
  MlxToTinygradCopyLeasePool,
  MlxToTinygradCopyLeasePools,
  MlxToTinygradLeaseKey,
  MlxToTinygradLeasePool,
  MlxToTinygradLeasePools,
)


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
    lazy = mx.arange(32, dtype=mx.float32) + 3
    pool = MlxToTinygradLeasePool.from_mlx(lazy, tg_dtype=dtypes.float32, capacity=1)

    with pool.acquire_from_mlx(lazy) as lease:
      np.testing.assert_array_equal(lease.tensor.numpy(), np.arange(32, dtype=np.float32) + np.float32(3))

  def test_raw_lease_tensor_can_observe_reused_slot_contents(self):
    first = mx.array(np.arange(16, dtype=np.float32), dtype=mx.float32)
    second = mx.array(np.arange(16, dtype=np.float32) + np.float32(10), dtype=mx.float32)
    pool = MlxToTinygradLeasePool.from_mlx(first, tg_dtype=dtypes.float32, capacity=1)

    lease_a = pool.acquire_from_mlx(first)
    raw_tensor = lease_a.tensor
    lease_a.release()

    with pool.acquire_from_mlx(second):
      np.testing.assert_array_equal(raw_tensor.numpy(), np.arange(16, dtype=np.float32) + np.float32(10))

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

  def test_safe_release_clears_owner_reference(self):
    array = mx.array(np.arange(8, dtype=np.float32), dtype=mx.float32)
    pool = MlxToTinygradLeasePool.from_mlx(array, tg_dtype=dtypes.float32, capacity=1)

    lease = pool.acquire_from_mlx(array)
    self.assertTrue(hasattr(pool._slots[0].borrower._base_buf, "_external_owner"))
    lease.release()
    self.assertFalse(hasattr(pool._slots[0].borrower._base_buf, "_external_owner"))

  def test_unsafe_release_without_sync_retains_owner_reference(self):
    array = mx.array(np.arange(8, dtype=np.float32), dtype=mx.float32)
    pool = MlxToTinygradLeasePool.from_mlx(array, tg_dtype=dtypes.float32, capacity=1, synchronize_on_release=False)

    lease = pool.acquire_from_mlx(array)
    lease.release(synchronize=False)
    self.assertTrue(hasattr(pool._slots[0].borrower._base_buf, "_external_owner"))

  def test_scoped_handoff_realizes_returned_tensor_before_release(self):
    first = mx.array(np.arange(16, dtype=np.float32), dtype=mx.float32)
    second = mx.array(np.arange(16, dtype=np.float32) + np.float32(5), dtype=mx.float32)
    pools = MlxToTinygradLeasePools(capacity_per_key=1, synchronize_on_release=True)

    result_first = pools.run_with_mlx_tensor(first, tg_dtype=dtypes.float32, fn=lambda t: (t + 1).sum())
    result_second = pools.run_with_mlx_tensor(second, tg_dtype=dtypes.float32, fn=lambda t: (t + 1).sum())

    np.testing.assert_allclose(result_first.numpy(), np.array((np.arange(16, dtype=np.float32) + 1).sum(), dtype=np.float32))
    np.testing.assert_allclose(result_second.numpy(), np.array((np.arange(16, dtype=np.float32) + 6).sum(), dtype=np.float32))

  def test_scoped_handoff_rejects_returning_alias_view(self):
    array = mx.array(np.arange(16, dtype=np.float32), dtype=mx.float32)
    pools = MlxToTinygradLeasePools(capacity_per_key=1, synchronize_on_release=True)

    with self.assertRaisesRegex(RuntimeError, "alias the borrowed slot"):
      pools.run_with_mlx_tensor(array, tg_dtype=dtypes.float32, fn=lambda t: t.reshape(2, 8))

  def test_scoped_handoff_rejects_stashed_lazy_tensor(self):
    array = mx.array(np.arange(16, dtype=np.float32), dtype=mx.float32)
    pools = MlxToTinygradLeasePools(capacity_per_key=1, synchronize_on_release=True)
    stash: dict[str, Tensor] = {}

    with self.assertRaisesRegex(RuntimeError, "leaked tensor\\(s\\) derived"):
      pools.run_with_mlx_tensor(
        array,
        tg_dtype=dtypes.float32,
        fn=lambda t: (stash.setdefault("u", t + 1), (t + 2).sum())[1],
      )

  def test_scoped_handoff_rejects_stashed_alias_view(self):
    array = mx.array(np.arange(16, dtype=np.float32), dtype=mx.float32)
    pools = MlxToTinygradLeasePools(capacity_per_key=1, synchronize_on_release=True)
    stash: dict[str, Tensor] = {}

    with self.assertRaisesRegex(RuntimeError, "leaked tensor\\(s\\) derived"):
      pools.run_with_mlx_tensor(
        array,
        tg_dtype=dtypes.float32,
        fn=lambda t: (stash.setdefault("u", t.reshape(2, 8)), (t + 2).sum())[1],
      )

  def test_scoped_handoff_rejects_returning_borrowed_tensor(self):
    array = mx.array(np.arange(8, dtype=np.float32), dtype=mx.float32)
    pool = MlxToTinygradLeasePool.from_mlx(array, tg_dtype=dtypes.float32, capacity=1)

    with self.assertRaisesRegex(RuntimeError, "borrowed tensor directly"):
      pool.run_with_mlx_tensor(array, lambda t: t)

  def test_keyed_lease_pools_bucket_by_contract(self):
    flat = mx.array(np.arange(16, dtype=np.float32), dtype=mx.float32)
    matrix = mx.array(np.arange(16, dtype=np.float32).reshape(4, 4), dtype=mx.float32)
    pools = MlxToTinygradLeasePools(capacity_per_key=2, synchronize_on_release=False)

    with pools.acquire_from_mlx(flat, tg_dtype=dtypes.float32) as flat_lease:
      self.assertEqual(flat_lease.key, MlxToTinygradLeaseKey((16,), "float32", dtypes.float32.base.name, 0))

    with pools.acquire_from_mlx(matrix, tg_dtype=dtypes.float32) as matrix_lease:
      self.assertEqual(matrix_lease.key, MlxToTinygradLeaseKey((4, 4), "float32", dtypes.float32.base.name, 0))

    self.assertEqual(pools.pool_count, 2)

  def test_alias_pool_registry_eviction_is_bounded(self):
    pools = MlxToTinygradLeasePools(capacity_per_key=1, max_pools=2, synchronize_on_release=True)
    arrays = [
      mx.array(np.arange(4, dtype=np.float32), dtype=mx.float32),
      mx.array(np.arange(6, dtype=np.float32).reshape(2, 3), dtype=mx.float32),
      mx.array(np.arange(8, dtype=np.float32).reshape(2, 2, 2), dtype=mx.float32),
    ]
    keys = [MlxToTinygradLeaseKey.from_storage(mx.metal._unsafe_export_storage(arr), tg_dtype=dtypes.float32) for arr in arrays]

    for arr in arrays:
      result = pools.run_with_mlx_tensor(arr, tg_dtype=dtypes.float32, fn=lambda t: (t + 1).sum())
      self.assertIsInstance(result, Tensor)

    self.assertEqual(pools.pool_count, 2)
    self.assertIsNone(pools.get_pool(keys[0]))
    self.assertIsNotNone(pools.get_pool(keys[1]))
    self.assertIsNotNone(pools.get_pool(keys[2]))

  def test_copy_pool_scoped_handoff_realizes_returned_tensor(self):
    first = mx.array(np.arange(16, dtype=np.float32), dtype=mx.float32)
    second = mx.array(np.arange(16, dtype=np.float32) + np.float32(3), dtype=mx.float32)
    pools = MlxToTinygradCopyLeasePools(capacity_per_key=1, synchronize_on_release=True)

    result_first = pools.run_with_mlx_tensor(first, tg_dtype=dtypes.float32, fn=lambda t: (t + 1).sum())
    result_second = pools.run_with_mlx_tensor(second, tg_dtype=dtypes.float32, fn=lambda t: (t + 1).sum())

    np.testing.assert_allclose(result_first.numpy(), np.array((np.arange(16, dtype=np.float32) + 1).sum(), dtype=np.float32))
    np.testing.assert_allclose(result_second.numpy(), np.array((np.arange(16, dtype=np.float32) + 4).sum(), dtype=np.float32))

  def test_copy_pool_registry_eviction_is_bounded(self):
    pools = MlxToTinygradCopyLeasePools(capacity_per_key=1, max_pools=2, synchronize_on_release=True)
    arrays = [
      mx.array(np.arange(4, dtype=np.float32), dtype=mx.float32),
      mx.array(np.arange(6, dtype=np.float32).reshape(2, 3), dtype=mx.float32),
      mx.array(np.arange(8, dtype=np.float32).reshape(2, 2, 2), dtype=mx.float32),
    ]
    keys = [MlxToTinygradCopyKey.from_storage(mx.metal._unsafe_export_storage(arr), tg_dtype=dtypes.float32) for arr in arrays]

    for arr in arrays:
      result = pools.run_with_mlx_tensor(arr, tg_dtype=dtypes.float32, fn=lambda t: (t + 1).sum())
      self.assertIsInstance(result, Tensor)

    self.assertEqual(pools.pool_count, 2)
    self.assertIsNone(pools.get_pool(keys[0]))
    self.assertIsNotNone(pools.get_pool(keys[1]))
    self.assertIsNotNone(pools.get_pool(keys[2]))

  def test_copy_pool_capacity_requires_release(self):
    array = mx.array(np.arange(16, dtype=np.float32), dtype=mx.float32)
    pool = MlxToTinygradCopyLeasePool.from_mlx(array, tg_dtype=dtypes.float32, capacity=1)

    lease = pool.acquire_from_mlx(array)
    with self.assertRaisesRegex(RuntimeError, "still in use"):
      pool.acquire_from_mlx(array)
    lease.release(synchronize=False)


if __name__ == "__main__":
  unittest.main()
