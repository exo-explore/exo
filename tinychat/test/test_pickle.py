import unittest, pickle
import numpy as np
from tinygrad import Tensor, TinyJit, Variable
from tinygrad.engine.schedule import create_schedule

class TestPickle(unittest.TestCase):
  def test_pickle_realized_tensor(self):
    t = Tensor.rand(10, 10).realize()
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  def test_pickle_unrealized_tensor(self):
    t = Tensor.ones(10, 10)
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  def test_pickle_variable(self):
    v = Variable("i", 1, 20).bind(10)
    t1 = Tensor.ones(10, v).contiguous()
    t2 = Tensor.ones(10, v).contiguous()
    ret = (t1+t2).sum(1)
    st = pickle.dumps(ret)
    del ret
    vt2 = pickle.loads(st)
    np.testing.assert_equal(vt2.numpy(), 20)

  def test_pickle_buffer_view(self):
    t = Tensor.arange(10, device="CLANG").contiguous().realize()
    vt = t[3:5].contiguous().realize()
    assert hasattr(vt.lazydata.buffer, 'base')
    ref_value = vt.tolist()
    st = pickle.dumps(vt)
    del t, vt
    vt2 = pickle.loads(st)
    assert hasattr(vt2.lazydata.buffer, 'base')
    assert ref_value == vt2.tolist()

  def test_pickle_numpy(self):
    t = Tensor(np.array([1,2,3,4.]))
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  def test_pickle_jit(self):
    @TinyJit
    def add(a, b): return a+b+1
    for _ in range(3): add(Tensor.rand(10, 10), Tensor.rand(10, 10))
    del add.fxn  # pickling the JIT requires the function to be deleted
    st = pickle.dumps(add)
    del add

    add_fxn = pickle.loads(st)
    x = Tensor.ones(10, 10).contiguous().realize()
    y = Tensor.ones(10, 10).contiguous().realize()
    print("post jit")
    out = add_fxn(x, y)
    np.testing.assert_equal(out.numpy(), 3)

  def test_pickle_schedule(self):
    a = Tensor([1,2])
    out = a + 2
    sched = create_schedule([out.lazydata])
    pk = pickle.dumps(sched)
    sched_pk = pickle.loads(pk)
    assert sched_pk[-1].ast == sched[-1].ast

if __name__ == '__main__':
  unittest.main()
