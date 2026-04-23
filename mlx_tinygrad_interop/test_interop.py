import unittest

import mlx.core as mx
import numpy as np
from tinygrad import Tensor, dtypes


@unittest.skipUnless(mx.metal.is_available(), "Metal is not available")
class TestMlxTinygradInterop(unittest.TestCase):
  def test_mlx_slice_import_uses_backing_buffer_bytes(self):
    backing = np.arange(4096, dtype=np.float32)
    base = mx.array(backing, dtype=mx.float32)
    view = base[16:1808]
    mx.eval(view)

    storage = mx.metal._unsafe_export_storage(view)
    self.assertNotIn("nbytes", storage)
    self.assertEqual(int(storage["offset_bytes"]), 64)
    self.assertEqual(int(storage["logical_nbytes"]), 7168)
    self.assertEqual(int(storage["buffer_nbytes"]), 16384)

    tensor = Tensor._unsafe_from_metal_buffer_fast(
      int(storage["mtl_buffer_ptr"]),
      tuple(storage["shape"]),
      dtype=dtypes.float32,
      byte_offset=int(storage["offset_bytes"]),
      buffer_nbytes=int(storage["buffer_nbytes"]),
      owner=view,
    )
    np.testing.assert_array_equal(tensor.numpy(), backing[16:1808])

  def test_single_entry_mlx_to_tinygrad_fast(self):
    values = np.arange(128, dtype=np.float32)
    array = mx.array(values, dtype=mx.float32)
    tensor = mx.metal._unsafe_to_tinygrad_fast(array, dtypes.float32, owner=array)
    np.testing.assert_array_equal(tensor.numpy(), values)

  def test_reusable_borrower_rebinds_between_sources(self):
    first = np.arange(64, dtype=np.float32)
    second = first + np.float32(1)
    array_a = mx.array(first, dtype=mx.float32)
    array_b = mx.array(second, dtype=mx.float32)
    storage_a = mx.metal._unsafe_export_storage(array_a)
    borrower = Tensor._unsafe_metal_borrower(
      int(storage_a["mtl_buffer_ptr"]),
      tuple(storage_a["shape"]),
      dtype=dtypes.float32,
      byte_offset=int(storage_a["offset_bytes"]),
      buffer_nbytes=int(storage_a["buffer_nbytes"]),
      owner=array_a,
    )

    tensor_a = mx.metal._unsafe_rebind_tinygrad(array_a, borrower, owner=array_a)
    tensor_b = mx.metal._unsafe_rebind_tinygrad(array_b, borrower, owner=array_b)

    self.assertIs(tensor_a, tensor_b)
    np.testing.assert_array_equal(tensor_a.numpy(), second)
    np.testing.assert_array_equal(tensor_b.numpy(), second)


if __name__ == "__main__":
  unittest.main()
