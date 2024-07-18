import unittest
from test.helpers import TestUOps
from tinygrad.dtype import dtypes
from tinygrad.ops import BinaryOps, TernaryOps
from tinygrad.codegen.uops import UOps, UOp
from tinygrad.codegen.uopgraph import UOpGraph, PatternMatcher, UPat, _match

class TestPatternMatcher(TestUOps):
  def test_simple_match(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x", dtype=dtypes.float), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_uop(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x"), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.ALU, dtypes.float, (c1, c1), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_uop_set(self):
    matcher = PatternMatcher([(UPat({UOps.CONST, UOps.CAST}, name="x"), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.bool, arg=False)
    c2 = UOp(UOps.CAST, dtypes.int, (c1,))
    c3 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c4 = UOp(UOps.ALU, dtypes.float, (c3, c3), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_arg(self):
    matcher = PatternMatcher([
      (UPat(UOps.CONST, 0, name="x"), lambda x: x),
      (UPat(UOps.CONST, False, name="x"), lambda x: x),
      (UPat(UOps.ALU, BinaryOps.MAX, name="x"), lambda x: x),
    ])
    c1 = UOp(UOps.CONST, dtypes.float, arg=0.0)
    c2 = UOp(UOps.CONST, dtypes.bool, arg=False)
    c3 = UOp(UOps.ALU, dtypes.float, (c1, c1), arg=BinaryOps.MAX)
    c4 = UOp(UOps.ALU, dtypes.float, (c1, c1), arg=BinaryOps.MUL)
    c5 = UOp(UOps.CONST, dtypes.int, arg=-1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)
    self.assertEqual(matcher.rewrite(c5), None)

  @unittest.skip("this is not supported any more")
  def test_arg_set(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, BinaryOps.MUL, (UPat(UOps.CONST, {-1, 1}), UPat(UOps.CONST, 2)), name="x"), lambda x: x)])
    y1 = UOp(UOps.CONST, dtypes.int, arg=1)
    y2 = UOp(UOps.CONST, dtypes.int, arg=2)
    y3 = UOp(UOps.CONST, dtypes.int, arg=-1)
    c1 = UOp(UOps.ALU, dtypes.int, (y1, y2), BinaryOps.MUL)
    c2 = UOp(UOps.ALU, dtypes.int, (y2, y2), BinaryOps.MUL)
    c3 = UOp(UOps.ALU, dtypes.int, (y3, y2), BinaryOps.MUL)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)
    self.assertEqual(matcher.rewrite(c3), c3)

  def test_dup_name(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST, name="y"), UPat(UOps.CONST, name="y"))), lambda x, y: x)])
    y1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    y2 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c1 = UOp(UOps.ALU, dtypes.float, (y1, y1), BinaryOps.ADD)
    c2 = UOp(UOps.ALU, dtypes.float, (y1, y2), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_dtype(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x", dtype=dtypes.float32), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float64, arg=1.0)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_dtype_set(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x", dtype=set([dtypes.float32, dtypes.float64])), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float64, arg=1.0)
    c3 = UOp(UOps.CONST, dtypes.float16, arg=1.0)
    c4 = UOp(UOps.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_vin_one(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST), UPat(UOps.CONST))), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c2), None)
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST), UPat(UOps.ALU))), lambda x: x)])
    c4 = UOp(UOps.ALU, dtypes.float, (c1,c3), BinaryOps.ADD)
    c5 = UOp(UOps.ALU, dtypes.float, (c3,c1), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), None)

  def test_vin_permutations(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=[UPat(UOps.CONST), UPat(UOps.ALU)]), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c4 = UOp(UOps.ALU, dtypes.float, (c3,c2), BinaryOps.ADD)
    c5 = UOp(UOps.ALU, dtypes.float, (c2,c3), BinaryOps.ADD)
    c6 = UOp(UOps.ALU, dtypes.float, (c3,c4), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), c5)
    self.assertEqual(matcher.rewrite(c6), None)

  def test_vin_repeat(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=UPat(UOps.CONST)), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c4 = UOp(UOps.ALU, dtypes.float, (c2,c3), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_allow_len(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST),), allow_any_len=True, arg=TernaryOps.MULACC), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.CONST, dtypes.float, arg=3.0)
    #c4 = UOp(UOps.ALU, dtypes.float, (c1,), UnaryOps.NEG)
    c5 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c6 = UOp(UOps.ALU, dtypes.float, (c1,c2,c3), TernaryOps.MULACC)
    #self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), None)
    self.assertEqual(matcher.rewrite(c6), c6)

  def test_deep_src_permutations(self):
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    u1 = (c1 + c2) + c1
    u2 = (c2 + c1) + c1
    pat = UPat(UOps.ALU, src = (UPat(UOps.ALU, src=[UPat(name='a'), UPat(name='b')]), UPat(name='b')))
    assert _match(u1, pat, {})
    assert _match(u2, pat, {})

  @unittest.skip("no longer supported")
  def test_rewrite_graph_folds(self):
    uops = UOpGraph()
    UOp(UOps.CONST, dtypes.float, arg=2.0, simplify=False)
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x", dtype=dtypes.float),
                               lambda x: UOp(UOps.CAST, dtypes.int, (UOp(UOps.ALU, x.dtype, (x, x), BinaryOps.ADD),)))])
    matcher.rewrite_graph(uops)
    # TODO: fix this. it's 2 now
    # self.assertEqual(len(uops.uops), 1)
    self.assertEqual(len(uops.uops), 2)
    self.assert_equiv_uops(UOp(UOps.CONST, dtypes.int, arg=4), uops.uops[-1])

  @unittest.skip("no longer supported")
  def test_rewrite_graph_adds(self):
    uops = UOpGraph()
    UOp(UOps.CONST, dtypes.int, arg=2, simplify=False)
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x", dtype=dtypes.int),
                               lambda x: UOp(UOps.STORE, x.dtype, (UOp(UOps.DEFINE_GLOBAL, x.dtype, tuple(), None), x)))])
    matcher.rewrite_graph(uops)
    uops.remove_childless(set(x for x in uops if x.op in {UOps.STORE}))

    self.assertEqual(len(uops.uops), 3)

    e1 = UOp(UOps.CONST, dtypes.int, arg=2)
    e2 = UOp(UOps.DEFINE_GLOBAL, dtypes.int, tuple())
    e3 = UOp(UOps.STORE, dtypes.int, (e2,e1))

    self.assert_equiv_uops(e1, uops.uops[0])
    self.assert_equiv_uops(e2, uops.uops[1])
    self.assert_equiv_uops(e3, uops.uops[2])

if __name__ == '__main__':
  unittest.main(verbosity=2)
