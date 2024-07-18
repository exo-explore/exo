#!/usr/bin/env python
import unittest
from tinygrad.tensor import Tensor
from tinygrad.ops import MetaOps, BufferOps
from tinygrad.nn import Conv2d
from tinygrad.engine.schedule import create_schedule

class TestConvShapetracker(unittest.TestCase):
  def test_conv_3x3_one_view(self):
    conv = Conv2d(16, 32, (3, 3))
    seen = set()

    # first run to init the weights, they are saved in seen
    create_schedule([conv(Tensor.empty(1, 16, 10, 10)).lazydata], seen)
    # run it again to get the kernels
    sched = [si for si in create_schedule([conv(Tensor.empty(1, 16, 10, 10)).lazydata], seen) if si.ast.op is MetaOps.KERNEL]
    assert len(sched) == 1, f"conv should only have one kernel, getting {len(sched)}"
    for st in [x.arg.st for x in sched[0].ast.lazyops if x.op is BufferOps.LOAD]:
      assert len(st.views) == 1

if __name__ == '__main__':
  unittest.main()
