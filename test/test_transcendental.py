import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.helpers import Context, getenv
from test.test_schedule import check_schedule
from test.test_dtype_alu import ht
from test.helpers import is_dtype_supported
import numpy as np
from hypothesis import given, settings, strategies as strat

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

class TestTranscendentalMath(unittest.TestCase):
  @unittest.skipUnless(is_dtype_supported(dtypes.float64, Device.DEFAULT), f"no float64 on {Device.DEFAULT}")
  @unittest.skipIf(getenv("CUDACPU") or (getenv("MOCKGPU") and Device.DEFAULT == "NV"), "crashed")
  @given(ht.float64, strat.sampled_from([(Tensor.exp, np.exp), (Tensor.log, np.log), (Tensor.sin, np.sin)]))
  def test_float64(self, x, op):
    if op[0] == Tensor.sin:
      # TODO: reduction does not work  # 536870912.125  # 2914593.01171875  # 134217728.03125
      if abs(x) > 536870912: return

    with Context(TRANSCENDENTAL=2):
      np.testing.assert_allclose(op[0](Tensor([x], dtype=dtypes.float64)).numpy(),
                                 op[1](np.array([x], dtype=_to_np_dtype(dtypes.float64))),
                                 atol=3e-2, rtol=1e-5)  # sin can have bigger atol for very big x

  @unittest.skipIf(getenv("CUDACPU") or (getenv("MOCKGPU") and Device.DEFAULT == "NV"), "crashed")
  @given(ht.float32, strat.sampled_from([(Tensor.exp, np.exp), (Tensor.log, np.log), (Tensor.sin, np.sin)]))
  def test_float32(self, x, op):
    with Context(TRANSCENDENTAL=2):
      np.testing.assert_allclose(op[0](Tensor([x], dtype=dtypes.float32)).numpy(),
                                 op[1](np.array([x], dtype=_to_np_dtype(dtypes.float32))),
                                 atol=2e-5, rtol=1e-5)

  @unittest.skipUnless(is_dtype_supported(dtypes.float16, Device.DEFAULT), f"no float16 on {Device.DEFAULT}")
  @given(ht.float16, strat.sampled_from([(Tensor.exp, np.exp), (Tensor.log, np.log), (Tensor.sin, np.sin)]))
  def test_float16(self, x, op):
    with Context(TRANSCENDENTAL=2):
      np.testing.assert_allclose(op[0](Tensor([x], dtype=dtypes.float16)).numpy(),
                                 op[1](np.array([x], dtype=_to_np_dtype(dtypes.float16))),
                                 atol=1e-2, rtol=4e-3)  # exp can have bigger rtol

class TestTranscendentalSchedule(unittest.TestCase):
  # w/ payne_hanek_reduction (fp32)
  def test_transcendental_sin_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.sin() + b.sin()
      c = c.sin()
      check_schedule(c, 1)

  def test_transcendental_log2_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.log2() + b.log2()
      c = c.log2()
      check_schedule(c, 1)

  def test_transcendental_exp2_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.exp2() + b.exp2()
      c = c.exp2()
      check_schedule(c, 1)

if __name__ == '__main__':
  unittest.main()
