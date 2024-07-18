#!/usr/bin/env python
import numpy as np
import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.lazy import LazyBuffer, ReduceOps, MetaOps
from tinygrad.engine.schedule import create_schedule

class TestLazyBuffer(unittest.TestCase):
  def test_fromcpu_shape_tracker(self):
    def helper(a: np.ndarray):
      print(a.shape, a.strides, a.flags.c_contiguous)
      b = Tensor(a).lazydata
      #assert b.st.contiguous == a.flags.c_contiguous
      assert b.st.shape == a.shape
      np.testing.assert_equal(a, Tensor(b).numpy())

    for ndims in range(1, 4):
      a = np.random.randn(*(4,)*ndims).astype(np.float32)
      for stride in [-2, 1, 2]:
        for start in [0, 1]:
          helper(a[(slice(start, None, stride),)*ndims])

  def test_shuffle_pad_ops_cmpeq(self):
    y = Tensor([1]).cat(Tensor([1]) == 0).numpy()
    z = Tensor([1, 0]).numpy()
    np.testing.assert_allclose(y, z)

  def test_shuffle_pad_ops_div(self):
    y = Tensor([1]).cat(Tensor([1]).div(Tensor([2.0]))).numpy()
    z = Tensor([1, 0.5]).numpy()
    np.testing.assert_allclose(y, z)

  def test_shuffle_pad_ops_log(self):
    y = Tensor([1]).cat(Tensor([1]).log()).numpy()
    z = Tensor([1, 0]).numpy()
    np.testing.assert_allclose(y, z)

  def test_shuffle_pad_ops_exp(self):
    y = Tensor([1]).cat(Tensor([1]).exp()).numpy()
    z = Tensor([1, np.e]).numpy()
    np.testing.assert_allclose(y, z)

  def test_device_0_is_the_same_device(self):
    a = Tensor([1, 2, 3], f"{Device.DEFAULT}")
    b = Tensor([1, 2, 3], f"{Device.DEFAULT}:0")
    assert a.device == b.device

  def test_shrink_const_into_zero(self):
    # regression test to make sure the shapetracker is preserved
    a = Tensor.zeros(4,4,4).shrink((None, (0,0), None))
    b = Tensor.zeros(4,1,4)
    c = a.cat(b, dim=1)
    np.testing.assert_allclose(c.numpy(), np.concatenate((a.numpy(), b.numpy()), axis=1))

  def test_shrink_const_then_cast(self):
    # regression test to make sure the shapetracker is preserved
    a = Tensor.zeros(4,4,4).shrink((None, (0,0), None)).cast(dtypes.int32)
    b = Tensor.zeros(4,1,4)
    c = a.cat(b, dim=1)
    np.testing.assert_allclose(c.numpy(), np.concatenate((a.numpy(), b.numpy()), axis=1))

  def test_const_dtype(self):
    lb: LazyBuffer = Tensor([1], dtype=dtypes.int).lazydata
    assert lb.const(1).base.arg == 1
    assert type(lb.const(1).base.arg) is int

    lb: LazyBuffer = Tensor([1], dtype=dtypes.float).lazydata
    assert lb.const(1).base.arg == 1.0
    assert type(lb.const(1).base.arg) is float

class TestReduceOp(unittest.TestCase):
  def test_no_split_reduce_kernel(self):
    a = Tensor.rand(4, 4).realize()
    a = a.sum()
    sched = create_schedule([a.lazydata])
    assert len(sched) == 1
    assert sched[0].ast.src[0].src[0].op is ReduceOps.SUM

  def test_split_reduce_kernel_dim0(self):
    a = Tensor.rand(256, 255).realize()
    a = a.sum()
    sched = create_schedule([a.lazydata])
    assert len(sched) == 2
    for s in sched:
      assert s.ast.src[0].src[0].op is ReduceOps.SUM

  def test_split_reduce_kernel_dim1(self):
    a = Tensor.rand(255, 256).realize()
    a = a.sum()
    sched = create_schedule([a.lazydata])
    assert len(sched) == 2
    for s in sched:
      assert s.ast.src[0].src[0].op is ReduceOps.SUM

class TestView(unittest.TestCase):
  def test_all_masked_out(self):
    # start with non CONST MetaOps
    a = Tensor.rand(10, 10)
    assert a.lazydata.base.op is not MetaOps.CONST

    # all masked out, degrades to const 0
    b = a.pad(((0, 10), None))[10:]
    assert b.shape == (10, 10)
    assert b.lazydata.base.op is MetaOps.CONST and b.lazydata.base.arg == 0

    # mask out dim = 1 works too
    b = a.pad((None, (0, 10)))[:, 10:]
    assert b.shape == (10, 10)
    assert b.lazydata.base.op is MetaOps.CONST and b.lazydata.base.arg == 0

    # partial masked out does not degrade into CONST
    b = a.pad(((0, 5), None))[5:]
    assert b.shape == (10, 10)
    assert b.lazydata.base.op is not MetaOps.CONST

if __name__ == "__main__":
  unittest.main()
