import unittest
from tinygrad import Tensor
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import lower_schedule_item
from tinygrad.codegen.uops import flops_mem, UOps, UOp
from tinygrad.codegen.uopgraph import UOpGraph
from tinygrad.ops import BinaryOps, TernaryOps
from tinygrad.dtype import dtypes

# TODO: can copy this in here when we remove it
#from tinygrad.ops import get_lazyop_info
#info = get_lazyop_info(ast)
#print(ops, mem, expected_mem)
#print(info.flops, info.mem_estimate)

# **************** new FlopCounter ****************

def get_stats(x:Tensor):
  si = create_schedule([x.lazydata])[-1]
  ei = lower_schedule_item(si)
  return ei.prg.op_estimate, ei.prg.mem_estimate

class TestUOpsStats(unittest.TestCase):
  def test_simple_add(self):
    a = Tensor.empty(100,100)
    b = Tensor.empty(100,100)
    c = a+b
    ops, mem = get_stats(c)
    expected_ops = c.numel()
    expected_mem = a.nbytes() + b.nbytes() + c.nbytes()
    self.assertEqual(mem, expected_mem)
    # NOTE; ops also include indexing ops
    assert expected_ops <= ops and ops <= expected_ops * 2

  def test_simple_add_sq(self):
    a = Tensor.empty(100,100)
    b = Tensor.empty(100,100)
    c = (a+b)*(a+b)
    ops, mem = get_stats(c)
    expected_ops = c.numel()*2
    expected_mem = a.nbytes() + b.nbytes() + c.nbytes()
    self.assertEqual(mem, expected_mem)
    # NOTE; ops also include indexing ops
    assert expected_ops <= ops and ops <= expected_ops * 2

  def test_simple_matmul(self):
    a = Tensor.empty(1024,1024)
    b = Tensor.empty(1024,1024)
    c = a@b
    ops, mem = get_stats(c)
    expected_ops = c.numel() * 1024 * 2
    required_mem = a.nbytes() + b.nbytes() + c.nbytes()
    assert expected_ops <= ops and ops <= expected_ops * 1.2
    # NOTE: it's hard to assert on the memory here, all depends on caching
    assert required_mem <= mem

  #MULACC should have the same stats as MUL + ADD
  def test_mulacc(self):
    globl = UOp(UOps.DEFINE_GLOBAL, dtypes.int, tuple())
    o1 = UOp(UOps.CONST, dtypes.int, tuple(), 1)
    o2 = UOp(UOps.CONST, dtypes.int, tuple(), 2)
    u1 = UOp(UOps.LOAD, dtypes.int, (globl, o1))
    u2 = UOp(UOps.LOAD, dtypes.int, (globl, o2))
    u3 = UOp(UOps.CONST, dtypes.int, tuple(), 3)
    u4 = UOp(UOps.ALU, dtypes.int, (u1,u2), BinaryOps.MUL)
    u5 = UOp(UOps.ALU, dtypes.int, (u4,u3), BinaryOps.ADD)
    uops = UOpGraph([u5])

    globl = UOp(UOps.DEFINE_GLOBAL, dtypes.int, tuple())
    o1 = UOp(UOps.CONST, dtypes.int, tuple(), 1)
    o2 = UOp(UOps.CONST, dtypes.int, tuple(), 2)
    u1 = UOp(UOps.LOAD, dtypes.int, (globl, o1))
    u2 = UOp(UOps.LOAD, dtypes.int, (globl, o2))
    u3 = UOp(UOps.CONST, dtypes.int, tuple(), 3)
    u4 = UOp(UOps.ALU, dtypes.int, (u1,u2,u3), TernaryOps.MULACC)
    uops_fma = UOpGraph([u4])

    self.assertEqual(flops_mem(uops.uops), flops_mem(uops_fma.uops))


if __name__ == '__main__':
  unittest.main(verbosity=2)
