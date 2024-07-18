#!/usr/bin/env python
import unittest
from tinygrad import dtypes, Tensor
from tinygrad.helpers import prod
from tinygrad.ops import LazyOp, UnaryOps, BinaryOps, ReduceOps, get_lazyop_info, BufferOps, MemBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.codegen.kernel import Kernel
from tinygrad.codegen.uops import flops_mem

class TestFlopCounter(unittest.TestCase):
  def setUp(self):
    self.buf0 = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float32, ShapeTracker.from_shape((4,))))
    self.buf1 = LazyOp(BufferOps.LOAD, (), MemBuffer(2, dtypes.float32, ShapeTracker.from_shape((4,))))
    self.buf2 = LazyOp(BufferOps.LOAD, (), MemBuffer(2, dtypes.float32, ShapeTracker.from_shape((4,4))))

  def compare_flop_counters(self, ast):
    info = get_lazyop_info(ast.src[0])
    lin = Kernel(ast)
    # NOTE: why does hand coded optimizations change flops for the GEMM?
    #lin.hand_coded_optimizations()
    lin.linearize()
    ops, mem = flops_mem(lin.uops.uops, ignore_indexing=True)
    run_count = prod((lin.global_size or []) + (lin.local_size or []))
    self.assertEqual(info.flops, ops*run_count)
    print(info.flops, info.mem_estimate, "vs", ops*run_count, mem*run_count)
    #lin.uops.print()

  def test_flops_sin(self):
    op0 = LazyOp(UnaryOps.SIN, (self.buf0,), None)
    info = get_lazyop_info(op0)
    self.assertEqual(info.flops, 4)

  def test_flops_add(self):
    op0 = LazyOp(BinaryOps.ADD, (self.buf0,self.buf1,), None)
    info = get_lazyop_info(op0)
    self.assertEqual(info.flops, 4)

  def test_flops_add_twice(self):
    op0 = LazyOp(BinaryOps.ADD, (self.buf0,self.buf1,), None)
    op1 = LazyOp(BinaryOps.ADD, (op0,self.buf1,), None)
    info = get_lazyop_info(op1)
    self.assertEqual(info.flops, 8)

  def test_flops_add_self(self):
    op0 = LazyOp(BinaryOps.ADD, (self.buf0,self.buf1,), None)
    op1 = LazyOp(BinaryOps.ADD, (op0,op0,), None)
    info = get_lazyop_info(op1)
    self.assertEqual(info.flops, 8)

  def test_flops_add_roundabout_self(self):
    op0 = LazyOp(BinaryOps.ADD, (self.buf0,self.buf1,), None)
    op1 = LazyOp(BinaryOps.ADD, (op0,self.buf1,), None)
    op2 = LazyOp(BinaryOps.ADD, (op0,op1,), None)
    info = get_lazyop_info(op2)
    self.assertEqual(info.flops, 12)

  def test_flops_red(self):
    op0 = LazyOp(BinaryOps.MUL, (self.buf0,self.buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (0,))
    op2 = LazyOp(BinaryOps.ADD, (op1, op1,), None)
    info = get_lazyop_info(op2)
    self.assertEqual(info.flops, 9)

  def test_flops_sum1d(self):
    op0 = LazyOp(ReduceOps.SUM, (self.buf0,), (0,))
    info = get_lazyop_info(op0)
    self.assertEqual(info.flops, 4)
    self.assertEqual(info.shape, (1,))

  def test_flops_sum2d(self):
    op0 = LazyOp(ReduceOps.SUM, (self.buf2,), (0,))
    info = get_lazyop_info(op0)
    self.assertEqual(info.flops, 16)
    self.assertEqual(info.shape, (1,4))

    op1 = LazyOp(ReduceOps.SUM, (op0,), (1,))
    info = get_lazyop_info(op1)
    self.assertEqual(info.flops, 16+4)
    self.assertEqual(info.shape, (1,1))

  def test_flops_conv(self):
    out = Tensor.empty(16,3,16,16).conv2d(Tensor.empty(64,3,3,3))
    self.compare_flop_counters(out.schedule()[-1].ast)

  def test_flops_gemm(self):
    out = Tensor.empty(4,16,16) @ Tensor.empty(4,16,16)
    self.compare_flop_counters(out.schedule()[-1].ast)

if __name__ == '__main__':
  unittest.main()
