import unittest
from tinygrad import Tensor, TinyJit, Variable, dtypes
import numpy as np

class TestSetitem(unittest.TestCase):
  def test_simple_setitem(self):
    cases = (
      ((6,6), (slice(2,4), slice(3,5)), Tensor.ones(2,2)),
      ((6,6), (slice(2,4), slice(3,5)), Tensor([1.,2.])),
      ((6,6), (slice(2,4), slice(3,5)), 1.0),
      ((6,6), (3, 4), 1.0),
      ((6,6), (3, None, 4, None), 1.0),
      ((4,4,4,4), (Ellipsis, slice(1,3), slice(None)), Tensor(4)),
      ((4,4,4,4), (Ellipsis, slice(1,3)), 4),
      ((4,4,4,4), (2, slice(1,3), None, 1), 4),
      ((4,4,4,4), (slice(1,3), slice(None), slice(0,4,2)), 4),
      ((4,4,4,4), (slice(1,3), slice(None), slice(None), slice(0,3)), 4),
      ((6,6), (slice(1,5,2), slice(0,5,3)), 1.0),
      ((6,6), (slice(5,1,-2), slice(5,0,-3)), 1.0),
    )
    for shp, slc, val in cases:
      t = Tensor.zeros(shp).contiguous()
      t[slc] = val
      n = np.zeros(shp)
      n[slc] = val.numpy() if isinstance(val, Tensor) else val
      np.testing.assert_allclose(t.numpy(), n)

  def test_setitem_into_unrealized(self):
    t = Tensor.arange(4).reshape(2, 2)
    t[1] = 5
    np.testing.assert_allclose(t.numpy(), [[0, 1], [5, 5]])

  def test_setitem_dtype(self):
    for dt in (dtypes.int, dtypes.float, dtypes.bool):
      for v in (5., 5, True):
        t = Tensor.ones(6,6, dtype=dt).contiguous()
        t[1] = v
        assert t.dtype == dt

  def test_setitem_into_noncontiguous(self):
    t = Tensor.ones(4)
    assert not t.lazydata.st.contiguous
    with self.assertRaises(AssertionError): t[1] = 5

  def test_setitem_inplace_operator(self):
    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] += 2
    np.testing.assert_allclose(t.numpy(), [[0, 1], [4, 5]])

    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] -= 1
    np.testing.assert_allclose(t.numpy(), [[0, 1], [1, 2]])

    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] *= 2
    np.testing.assert_allclose(t.numpy(), [[0, 1], [4, 6]])

    # NOTE: have to manually cast setitem target to least_upper_float for div
    t = Tensor.arange(4, dtype=dtypes.float).reshape(2, 2).contiguous()
    t[1] /= 2
    np.testing.assert_allclose(t.numpy(), [[0, 1], [1, 1.5]])

    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] **= 2
    np.testing.assert_allclose(t.numpy(), [[0, 1], [4, 9]])

    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] ^= 5
    np.testing.assert_allclose(t.numpy(), [[0, 1], [7, 6]])

  @unittest.expectedFailure
  def test_setitem_consecutive_inplace_operator(self):
    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] += 2
    t = t.contiguous()
    # TODO: RuntimeError: must be contiguous for assign ShapeTracker(views=(View(shape=(2,), strides=(1,), offset=2, mask=None, contiguous=False),))
    t[1] -= 1
    np.testing.assert_allclose(t.numpy(), [[0, 1], [3, 4]])

  # TODO: implement fancy setitem
  @unittest.expectedFailure
  def test_fancy_setitem(self):
    t = Tensor.zeros(6,6).contiguous()
    t[[1,2], [3,2]] = 3
    n = np.zeros((6,6))
    n[[1,2], [3,2]] = 3
    np.testing.assert_allclose(t.numpy(), n)

  def test_simple_jit_setitem(self):
    @TinyJit
    def f(t:Tensor, a:Tensor):
      t[2:4, 3:5] = a

    for i in range(1, 6):
      t = Tensor.zeros(6, 6).contiguous().realize()
      a = Tensor.full((2, 2), fill_value=i, dtype=dtypes.float).contiguous()
      f(t, a)

      n = np.zeros((6, 6))
      n[2:4, 3:5] = np.full((2, 2), i)
      np.testing.assert_allclose(t.numpy(), n)

  def test_jit_setitem_variable_offset(self):
    @TinyJit
    def f(t:Tensor, a:Tensor, v:Variable):
      t.shrink(((v,v+1), None)).assign(a).realize()

    t = Tensor.zeros(6, 6).contiguous().realize()
    n = np.zeros((6, 6))

    for i in range(6):
      v = Variable("v", 0, 6).bind(i)
      a = Tensor.full((1, 6), fill_value=i+1, dtype=dtypes.float).contiguous()
      n[i, :] = i+1
      f(t, a, v)
      np.testing.assert_allclose(t.numpy(), n)
    np.testing.assert_allclose(t.numpy(), [[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4],[5,5,5,5,5,5],[6,6,6,6,6,6]])

class TestWithGrad(unittest.TestCase):
  def test_no_requires_grad_works(self):
    z = Tensor.rand(8, 8)
    x = Tensor.rand(8)
    z[:3] = x

  def test_set_into_requires_grad(self):
    z = Tensor.rand(8, 8, requires_grad=True)
    x = Tensor.rand(8)
    with self.assertRaises(NotImplementedError):
      z[:3] = x

  def test_set_with_requires_grad(self):
    z = Tensor.rand(8, 8)
    x = Tensor.rand(8, requires_grad=True)
    with self.assertRaises(NotImplementedError):
      z[:3] = x

if __name__ == '__main__':
  unittest.main()