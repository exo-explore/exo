import unittest
import time
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import lower_schedule_item, run_schedule

class TestFusionOp(unittest.TestCase):
  def test_contiguous_add(self):
    def test(contig=False):
      bt = Tensor(np.arange(16), dtype=dtypes.float32).reshape(4,4)
      x = bt.permute(1,0)
      if contig: x = x.contiguous()
      return (x.permute(1,0) + bt).data()
    assert test() == test(True)

  def test_expand_fuse(self):
    bt = Tensor(np.ones((10, 1)), dtype=dtypes.float32)
    out = (bt*2).expand(10,10).sum(1)
    sched = create_schedule([out.lazydata], None)
    run_schedule(sched)
    outd = out.tolist()
    assert all(x == 20.0 for x in outd)

  def test_recursive_add(self):
    st = time.perf_counter()
    a = Tensor([1,2,3,4])
    for _ in range(24): a = a + a
    sched = create_schedule([a.lazydata], None)
    ei = lower_schedule_item(sched[-1])
    self.assertLess(time.perf_counter()-st, 1.0)
    assert len(ei.prg.p.src.splitlines()) < 250

  def test_recursive_add_cmp(self):
    st = time.perf_counter()
    a = Tensor([1,2,3,4])
    for _ in range(24): a = a + a
    sched1 = create_schedule([a.lazydata], None)
    b = Tensor([1,2,3,4])
    for _ in range(24): b = b + b
    sched2 = create_schedule([b.lazydata], None)
    c = Tensor([1,2,3,4])
    for _ in range(23): c = c + c
    sched3 = create_schedule([c.lazydata], None)
    assert sched1[-1].ast == sched2[-1].ast
    assert sched1[-1].ast != sched3[-1].ast
    self.assertLess(time.perf_counter()-st, 1.0)

if __name__ == '__main__':
  unittest.main(verbosity=2)
