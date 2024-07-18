from typing import List, Tuple, Dict, Union
import numpy as np
import unittest
from dataclasses import replace
from test.external.fuzz_linearizer import compare_linearizer

from tinygrad.codegen.kernel import Opt, OptOps, KernelOptError, Kernel
from tinygrad.codegen.lowerer import get_grouped_dims
from tinygrad.codegen.uops import UOp, UOps
from tinygrad.device import Device, Buffer
from tinygrad.ops import BinaryOps, BufferOps, MemBuffer, ConstBuffer, LazyOp, MetaOps, TernaryOps, ReduceOps, UnaryOps
from tinygrad.renderer import TensorCore
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import run_schedule, lower_schedule, CompiledRunner
from tinygrad.engine.graph import print_tree
from tinygrad.helpers import DEBUG, prod, Context, getenv, CI, flatten, dedup
from tinygrad.dtype import DType, dtypes

def helper_realized_ast(r:Union[Tensor, List[Tensor]]):
  if isinstance(r, Tensor): r = [r]
  s = create_schedule([x.lazydata for x in r])
  run_schedule(s[:-1])  # run all kernels except the last one
  # now all input LazyBuffers buffers in s[-1] should be realized
  # allocate an output buffer
  output_buffers = [Buffer((out).device, out.size, out.dtype).allocate() for out in s[-1].outputs]
  return s[-1].ast, output_buffers+list(s[-1].inputs)

def helper_tc_allclose(n:int, m:int, k:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_opt:int=0):
  a, b = Tensor.rand(m, k, dtype=dtype_in), Tensor.rand(k, n, dtype=dtype_in)
  np_a, np_b = a.numpy(), b.numpy()
  r = a.matmul(b, acc_dtype=dtype_out)
  sched = create_schedule([r.lazydata])
  realized_ast = sched[-1].ast
  run_schedule(sched)
  out = r.numpy()
  k = Kernel(realized_ast)
  k.apply_tensor_cores(1, axis=axis, tc_opt=tc_opt)
  k.linearize()
  assert len([uop for uop in k.uops if uop.op is UOps.WMMA]) > 0, "tensor core not triggered"
  assert len([x for x in k.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"
  np_c = np_a @ np_b
  if dtype_out == dtypes.half: tc_atol, tc_rtol = 1e-2, 1e-3
  elif dtype_in == dtypes.bfloat16: tc_atol, tc_rtol = 1e-2, 3e-3
  else: tc_atol, tc_rtol = 5e-3, 1e-4
  np.testing.assert_allclose(np_c, out, atol=tc_atol, rtol=tc_rtol)

def helper_tc_ensure_uops_and_opts_count(n: int, m:int, k:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_opt:int=0, ensure_triggered:bool=True):
  a, b = Tensor.rand(m, k, dtype=dtype_in), Tensor.rand(k, n, dtype=dtype_in)
  r = a.matmul(b, acc_dtype=dtype_out)
  sched = create_schedule([r.lazydata])
  realized_ast = sched[-1].ast
  k = Kernel(realized_ast)
  k.apply_tensor_cores(1, axis=axis, tc_opt=tc_opt)
  k.linearize()
  wmmas = len([uop for uop in k.uops if uop.op is UOps.WMMA])
  tcs = len([x for x in k.applied_opts if x.op is OptOps.TC])
  if ensure_triggered:
    assert wmmas > 0, "tensor core not triggered"
    assert tcs == 1, "tensor core opt not included"
  else:
    assert wmmas == 0, "tensor core is incorrectly triggered"
    assert tcs == 0, "tensor core opt is incorrectly included"

class TestLinearizer(unittest.TestCase):
  def test_arg_dedup(self):
    a, b = Tensor.randn(4), Tensor.randn(4)
    np_a, np_b = a.numpy(), b.numpy()
    c = ((a.shrink(((0, 2),)) - a.shrink(((2, 4),))) - (b.shrink(((0, 2),)) - b.shrink(((2, 4),))))
    lowered = list(lower_schedule(create_schedule([c.lazydata])))
    for ei in lowered: ei.run()
    rawbufs = lowered[-1].bufs
    assert len(rawbufs) == 3 and set(rawbufs[1:]) == {a.lazydata.base.realized, b.lazydata.base.realized}
    np_c = (np_a[:2] - np_a[2:]) - (np_b[:2] - np_b[2:])
    np.testing.assert_allclose(np_c, c.numpy(), atol=1e-4, rtol=1e-4)

  def test_load_removed(self):
    a = Tensor.rand(1).realize()
    b = Tensor.rand(1).realize()
    ta = Tensor.where(Tensor(True), a, b).numpy()
    tb = Tensor.where(Tensor(False), a, b).numpy()
    np.testing.assert_equal(a.numpy(), ta)
    np.testing.assert_equal(b.numpy(), tb)

  def test_multioutput(self):
    dtype, st = dtypes.int, ShapeTracker.from_shape((8,))
    a = LazyOp(BufferOps.LOAD, arg=MemBuffer(idx=2, dtype=dtype, st=st))
    b = LazyOp(BufferOps.LOAD, arg=MemBuffer(idx=3, dtype=dtype, st=st))
    out0 = LazyOp(BufferOps.STORE, (LazyOp(op=BinaryOps.ADD, src=(a,b)),), MemBuffer(idx=0, dtype=dtype, st=st))
    out1 = LazyOp(BufferOps.STORE, (LazyOp(op=BinaryOps.MUL, src=(a,b)),), MemBuffer(idx=1, dtype=dtype, st=st))

    a_t = Tensor.full(st.shape, 2).contiguous().realize()
    b_t = Tensor.full(st.shape, 3).contiguous().realize()
    lin = helper_linearizer_ast((out0, out1), [a_t, b_t], wanna_output=[a_t.numpy()+b_t.numpy(), a_t.numpy()*b_t.numpy()])[0]

    stores = [u for u in lin.uops if u.op is UOps.STORE]
    mutable_bufs = [u for u in lin.uops if u.op is UOps.DEFINE_GLOBAL and u.arg[-1]]
    assert len(mutable_bufs) == len(stores) == 2
    assert [u.arg[0] for u in mutable_bufs] == [0, 1]

  @unittest.skip("TODO: fix uops toposort")
  def test_sum_multireduce(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(32, dtype=dtypes.float).realize()
    first_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((1, 32)).expand((32, 32))))
    first_reduce = LazyOp(ReduceOps.SUM, (first_x,), (1,))
    second_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((32, 1))))
    squares = (second_x-first_reduce)
    squares_sum = LazyOp(ReduceOps.SUM, (squares,), (0,))
    store = LazyOp(BufferOps.STORE, (squares_sum,), MemBuffer(0, dtypes.float, ShapeTracker.from_shape((1, 1))))
    wanna_output = (x.numpy()-x.numpy().sum(-1, keepdims=True)).sum(-1)
    helper_linearizer_ast((store, ), [x], wanna_output=[wanna_output])

  @unittest.skip("TODO: fix uops toposort")
  def test_double_sum_multireduce(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(2, 32, 4, 16, dtype=dtypes.float).realize()
    first_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((2, 1, 32, 4, 1, 16)).expand((2, 32, 32, 4, 16, 16))))
    first_reduce = LazyOp(ReduceOps.SUM, (first_x,), (2,5))
    second_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((2, 32, 1, 4, 16, 1))))
    squares = (second_x-first_reduce)
    squares_sum = LazyOp(ReduceOps.SUM, (squares,), (1,4))
    store = LazyOp(BufferOps.STORE, (squares_sum,), MemBuffer(0, dtypes.float, ShapeTracker.from_shape((2, 1, 1, 4, 1, 1))))
    wanna_output = (x.numpy()-x.numpy().sum(axis=(1,3), keepdims=True)).sum(axis=(1,3)).reshape((2,1,1,4,1,1))
    helper_linearizer_ast((store, ), [x], wanna_output=[wanna_output])

  @unittest.skipIf(CI and Device.DEFAULT in {"PTX", "AMD", "NV"}, "ocelot/remu doesn't have multiple wave syncs yet")
  @unittest.skip("TODO: fix uops toposort")
  def test_var_multireduce(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(3, 27, 32, dtype=dtypes.float).realize()
    # push reduce (3, 27, 32) -> (3, 27, 1) -> (3, 27, 32) expand to LOAD
    first_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((3, 27, 1, 32)).expand((3, 27, 32, 32))))
    first_reduce = LazyOp(ReduceOps.SUM, (first_x,), (3,))
    mean = first_reduce * LazyOp(BufferOps.CONST, (), ConstBuffer(0.03125, dtypes.float, ShapeTracker.from_shape(()).reshape((1, 1, 1, 1)).expand((3, 27, 32, 1)))) # noqa: E501
    # store = LazyOp(BufferOps.STORE, (mean,), MemBuffer(0, dtypes.float, ShapeTracker.from_shape((3, 27, 32, 1))))
    # verify_lazyop(store)
    second_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((3, 27, 32, 1))))
    squares = (second_x-mean)*(second_x-mean)
    squares_sum = LazyOp(ReduceOps.SUM, (squares,), (2,))
    variance = squares_sum * LazyOp(BufferOps.CONST, (), ConstBuffer(0.03125, dtypes.float, ShapeTracker.from_shape(()).reshape((1, 1, 1, 1)).expand((3, 27, 1, 1)))) # noqa: E501
    store = LazyOp(BufferOps.STORE, (variance,), MemBuffer(0, dtypes.float, ShapeTracker.from_shape((3, 27, 1, 1))))
    wanna_output = x.numpy().var(axis=2, ddof=0).reshape((3,27,1,1))
    helper_linearizer_ast((store, ), [x], wanna_output=[wanna_output])
    # tinygrad ref
    y_tiny = x.var(axis=2, correction=0).reshape(3,27,1,1)
    np.testing.assert_allclose(y_tiny.numpy(), wanna_output, atol=1e-4, rtol=1e-4)

  # *** buildup to fused indexing
  @unittest.skipIf(CI, "very slow because of recomputing")
  def test_arange_expanded(self):
    # Tensor.arange(16384) expanded such that output shape is (4, 16384, 256, 1)
    # basically it's pushing the expand through this reduce:
    tiny = Tensor.arange(16384).reshape(16384, 1).expand(4, 16384, 256).reshape(4, 16384, 256, 1)
    real_arange = np.broadcast_to(np.arange(16384).reshape(16384, 1), (4, 16384, 256)).reshape(4, 16384, 256, 1)
    # NOTE: this is stupidly recomputing because it's not fused, but it proves a point.
    arange_input_st = ShapeTracker(views=(View(shape=(16385, 32767), strides=(0, 0), offset=0, mask=((0, 16385), (16383, 32767)), contiguous=False), \
        View(shape=(16384, 16384), strides=(1, 32768), offset=0, mask=None, contiguous=False)))
    arange_input_st = arange_input_st.reshape((1, 16384, 1, 16384)).expand((4, 16384, 256, 16384))
    arange_axis = (3,)
    arange = LazyOp(ReduceOps.SUM, (LazyOp(BufferOps.CONST, (), ConstBuffer(1, dtypes.int, arange_input_st)), ), arange_axis)
    output_shape = tuple(1 if i in arange_axis else s for i,s in enumerate(arange_input_st.shape))
    out = arange-LazyOp.const(1, dtypes.int, output_shape)
    store = LazyOp(BufferOps.STORE, (out, ), MemBuffer(0, dtypes.int, st=ShapeTracker.from_shape(output_shape)))
    helper_linearizer_ast((store, ), [], wanna_output=[real_arange])
    with Context(DEBUG=0, NOOPT=0): np.testing.assert_equal(tiny.numpy(), real_arange)

  @unittest.skipIf(CI and Device.DEFAULT in {"PTX", "AMD", "NV"}, "very slow")
  def test_indexing_multireduce(self):
    arange_input_st = ShapeTracker(views=(View(shape=(16385, 32767), strides=(0, 0), offset=0, mask=((0, 16385), (16383, 32767)), contiguous=False), \
        View(shape=(16384, 16384), strides=(1, 32768), offset=0, mask=None, contiguous=False)))
    # TODO: do this arange broadcast in the scheduler
    arange_input_st = arange_input_st.reshape((1, 16384, 1, 16384)).expand((4, 16384, 256, 16384))
    arange_axis = (3,)
    arange = LazyOp(ReduceOps.SUM, (LazyOp(BufferOps.CONST, (), ConstBuffer(1, dtypes.int, arange_input_st)), ), arange_axis)
    arange_out_shape = tuple(1 if i in arange_axis else s for i,s in enumerate(arange_input_st.shape))
    arange = arange-LazyOp.const(1, dtypes.int, arange_out_shape)
    # p2: the indexing
    dataset = Tensor.rand(16384, 256).realize()
    data1 = MemBuffer(1, dataset.dtype, ShapeTracker.from_shape(dataset.shape).reshape((1, 16384, 256, 1)).expand(arange_out_shape))
    idxs = Tensor([0,3,5,6]).realize()
    data2 = MemBuffer(2, dtypes.int, ShapeTracker.from_shape((4,)+(1,)*(len(arange_out_shape)-1)).expand(arange_out_shape))
    reduce_input = LazyOp(BufferOps.LOAD, (), data1)*LazyOp(UnaryOps.CAST, (arange.eq(LazyOp(BufferOps.LOAD, (), data2)),), dataset.dtype)
    out = LazyOp(ReduceOps.SUM, (reduce_input, ), (1,))
    output_shape = tuple(1 if i in out.arg else s for i,s in enumerate(arange_out_shape))
    store = LazyOp(BufferOps.STORE, (out, ), MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape(output_shape)))
    real_index = dataset.numpy()[idxs.numpy()].reshape(4, 1, 256, 1)
    helper_linearizer_ast((store, ), [dataset, idxs], wanna_output=[real_index])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_end_local(self):
    load = MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker.from_shape((32,)))
    store = MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker.from_shape((1,)))
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, arg=load),), arg=(0,)),), arg=store),

    load_t = Tensor.full(load.st.shape, 1).contiguous().realize()
    k = helper_linearizer_ast(ast, [load_t], wanna_output=[load_t.numpy().sum()])[1]
    self.assertEqual(k.uops[-1].op, UOps.ENDIF)
    self.assertLess(k.uops.uops.index([x for x in k.uops.uops if x.op is UOps.STORE][-1]), k.uops.uops.index(k.uops[-1]))

  def test_two_nested_range(self):
    a = Tensor.randn(2, ).realize()
    out = a.reshape(2, 1).expand(2, 3).sum()
    lin = helper_linearizer_opt(out, wanna_output=[np.broadcast_to(a.numpy().reshape(2, 1), (2, 3)).sum()])[0]
    ranges = [i for i,u in enumerate(lin.uops) if u.op is UOps.RANGE]
    # RANGE -> LOAD -> RANGE -> PHI
    assert any(x.op is UOps.LOAD for x in lin.uops[ranges[0]:ranges[1]])

  def test_three_nested_range(self):
    a = Tensor.randn(2, ).realize()
    out = a.reshape(2, 1).expand(2, 3).expand(2, 2, 3).sum()
    lin = helper_linearizer_opt(out, wanna_output=[np.broadcast_to(np.broadcast_to(a.numpy().reshape(2, 1), (2, 3)), (2, 2, 3)).sum()])[0]
    ranges = [i for i,u in enumerate(lin.uops) if u.op is UOps.RANGE]
    # RANGE -> RANGE -> LOAD -> RANGE -> PHI
    # NOTE: nothing should toposort between the first two ranges
    assert ranges[0]+1 == ranges[1]
    assert any(x.op is UOps.LOAD for x in lin.uops[ranges[1]:ranges[2]])

  def test_two_nested_range_alt_indexing(self):
    a = Tensor([2, 2]).realize()
    out = a.reshape(2, 1).pad(((1, 1), (1, 1)), 2).sum()
    lin = helper_linearizer_opt(out, wanna_output=[24])[0]
    ranges = [i for i,u in enumerate(lin.uops) if u.op is UOps.RANGE]
    # RANGE -> ALU -> RANGE -> ALU + LOAD -> PHI
    assert any(x.op is UOps.ALU for x in lin.uops[ranges[0]:ranges[1]])
    assert not any(x.op is UOps.LOAD for x in lin.uops[ranges[0]:ranges[1]])
    assert any(x.op in {UOps.ALU, UOps.LOAD} for x in lin.uops[ranges[1]:])

  def test_range_outer_op_before_phi(self):
    a = Tensor.randn(4, 1).realize()
    b = Tensor.randn(1, 1).realize()
    out = (a + b[0]).sum() + b[0]
    lin = helper_linearizer_opt(out, wanna_output=[(a.numpy()+b.numpy()[0]).sum()+b.numpy()])[0]
    ranges = [i for i,u in enumerate(lin.uops) if u.op is UOps.RANGE]
    # LOAD -> RANGE -> LOAD -> PHI
    assert lin.uops[ranges[0]-2].op is UOps.LOAD

  # TODO: this test is brittle
  def test_range_outer_op_before_phi_nested_range(self):
    a = Tensor.randn(2, ).realize()
    b = Tensor.randn(1, 1).realize()
    out = (a.reshape(2, 1).expand(2, 3) + b[0]).sum() + b[0]
    lin = helper_linearizer_opt(out, wanna_output=[(np.broadcast_to(a.numpy().reshape(2, 1), (2, 3)) + b.numpy()[0]).sum() + b.numpy()])[0]
    ranges = [i for i,u in enumerate(lin.uops) if u.op is UOps.RANGE]
    if getenv("PTX"):
    # LOAD -> RANGE -> CAST -> ALU -> ALU -> LOAD -> ALU -> RANGE -> ALU -> PHI
      assert lin.uops[ranges[0]-2].op is UOps.LOAD
      assert ranges[1] == ranges[0]+6
      assert [x.op for x in lin.uops[ranges[1]-2:ranges[1]]] == [UOps.LOAD, UOps.ALU]
    # LOAD -> RANGE -> LOAD -> ALU -> RANGE -> PHI
    else:
      assert lin.uops[ranges[0]-2].op is UOps.LOAD
      assert ranges[1] == ranges[0]+3
      assert [x.op for x in lin.uops[ranges[1]-2:ranges[1]]] == [UOps.LOAD, UOps.ALU]

  def test_range_outer_op_after_phi(self):
    a = Tensor.randn(4, 1).realize()
    out = a.sum() * a.sum()
    lin = helper_linearizer_opt(out, wanna_output=[a.numpy().sum()*a.numpy().sum()])[0]
    # RANGE -> LOAD -> PHI -> ALU
    end = max(i for i,u in enumerate(lin.uops) if u.op is UOps.ENDRANGE)
    assert lin.uops[end+1].op is UOps.ALU

  def test_range_outer_op_after_phi_nested_range(self):
    a = Tensor.randn(2, ).realize()
    out = a.reshape(2, 1).expand(2, 3).sum() + a.reshape(2, 1).expand(2, 3).sum()
    lin = helper_linearizer_opt(out, wanna_output=[(np.broadcast_to(a.numpy().reshape(2, 1), (2, 3))).sum()*2])[0]
    # RANGE -> LOAD -> PHI -> ALU
    end = max(i for i,u in enumerate(lin.uops) if u.op is UOps.ENDRANGE)
    assert lin.uops[end+1].op is UOps.ALU

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skip("AST has implicit movement ops")
  def test_early_end_local(self):
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=UnaryOps.SQRT, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=UnaryOps.NEG, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))),), arg=(2,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.03125, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=None)), arg=None), LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=UnaryOps.NEG, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))),), arg=(2,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.03125, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=None)), arg=None)), arg=None),), arg=(2,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.03125, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 1), strides=(27, 1, 0), offset=0, mask=None, contiguous=True),))))), arg=None),), arg=None),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 1), strides=(27, 1, 0), offset=0, mask=None, contiguous=True),)))), # noqa: E501
    k = Kernel(ast)
    k.hand_coded_optimizations()
    k.linearize()
    self.assertEqual(len(endifs:=[x for x in k.uops if x.op is UOps.ENDIF]), len(ifs:=[x for x in k.uops if x.op is UOps.IF]))
    self.assertEqual(len(barriers:=[x for x in k.uops if x.op is UOps.BARRIER]), 3)
    self.assertEqual(k.uops[k.uops.uops.index(endifs[0])-1].op, UOps.STORE)
    self.assertEqual(k.uops[k.uops.uops.index(endifs[0])+1], barriers[1])
    self.assertEqual(k.uops[k.uops.uops.index(endifs[0])+2].op, UOps.LOAD)
    self.assertLess(k.uops.uops.index(barriers[0]), k.uops.uops.index(ifs[0]))
    self.assertLess(k.uops.uops.index(ifs[0]), k.uops.uops.index(endifs[0]))
    self.assertLess(k.uops.uops.index(barriers[1]), k.uops.uops.index(ifs[1]))
    x = Tensor.randn(3,27,32).realize()
    helper_linearizer_ast(ast, [x], wanna_output=[x.numpy().std(axis=2, ddof=0).reshape(-1)])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skip("AST has implicit movement ops")
  def test_reduceops_order(self):
    # make sure that the kernel put reduceops in the order of their dependencies when passed to the Linearizer in arbitrary order
    load = MemBuffer(idx=4, dtype=dtypes.float, st=ShapeTracker.from_shape((32,)))
    ast0 = LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=load),), arg=(0,))
    ast1 = LazyOp(op=ReduceOps.SUM, src=(LazyOp(BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=load), \
      LazyOp(op=UnaryOps.NEG, src=(ast0,), arg=None))),), arg=(0,))
    ast2 = LazyOp(op=ReduceOps.SUM, src=(LazyOp(BinaryOps.ADD, src=(ast1, LazyOp(op=UnaryOps.NEG, \
      src=(LazyOp(op=BufferOps.LOAD, src=(), arg=load),), arg=None))),), arg=(0,))
    ast3 = LazyOp(op=ReduceOps.SUM, src=(LazyOp(BinaryOps.ADD, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=load), LazyOp(op=UnaryOps.NEG, src=(ast2,), arg=None))), LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=load), LazyOp(op=UnaryOps.NEG, src=(ast0,), arg=None))),)),), arg=(0,)) # noqa E501
    for order in [(d, c, b, a) for d in range(4) for c in range(4) for b in range(4) for a in range(4) if len(set([a,b,c,d])) == 4]:
      asts = [
        LazyOp(op=BufferOps.STORE, src=(ast0,), arg=MemBuffer(idx=order.index(0), dtype=dtypes.float, st=ShapeTracker.from_shape((1,)))),
        LazyOp(op=BufferOps.STORE, src=(ast1,), arg=MemBuffer(idx=order.index(1), dtype=dtypes.float, st=ShapeTracker.from_shape((1,)))),
        LazyOp(op=BufferOps.STORE, src=(ast2,), arg=MemBuffer(idx=order.index(2), dtype=dtypes.float, st=ShapeTracker.from_shape((1,)))),
        LazyOp(op=BufferOps.STORE, src=(ast3,), arg=MemBuffer(idx=order.index(3), dtype=dtypes.float, st=ShapeTracker.from_shape((1,))))
      ]
      k = Kernel([asts[i] for i in order])
      def recursive_reduceops(x: LazyOp): return [c for v in x.src for c in recursive_reduceops(v)] + [v for v in list(x.src) if v.op in ReduceOps]
      for i,r in enumerate(k.reduceops): assert not any([r in recursive_reduceops(x) for x in k.reduceops[:i]]), "reduceops are out of order"
      x = Tensor.randn(32).realize()
      outs = [b:=(a:=x.numpy()).sum(), c:=(a - b).sum(), d:=(c - a).sum(), (a-d + a-b).sum()]
      helper_linearizer_ast(tuple(asts[i] for i in order), [x], wanna_output=[outs[i] for i in order])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skip("AST has implicit movement ops")
  def test_multireduce_store_locals(self):
    # ensure the result of local reducop is stored and loaded back into every thread for future use
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=UnaryOps.SQRT, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=UnaryOps.NEG, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))),), arg=(2,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.03125, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=None)), arg=None), LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=UnaryOps.NEG, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))),), arg=(2,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.03125, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=None)), arg=None)), arg=None),), arg=(2,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.03125, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 1), strides=(27, 1, 0), offset=0, mask=None, contiguous=True),))))), arg=None),), arg=None),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 1), strides=(27, 1, 0), offset=0, mask=None, contiguous=True),)))), # noqa: E501
    k = Kernel(ast)
    k.hand_coded_optimizations()
    k.linearize()
    local_buf = [u for u in k.uops if u.op is UOps.DEFINE_LOCAL]
    self.assertEqual(len(real_local_stores:=[u for u in k.uops if u.op is UOps.STORE and any([lb in u.src for lb in local_buf])]), 3, \
      f"should have generated 3 BufferOps.STORE to the local buf but got {len(real_local_stores)}")
    self.assertEqual(len(real_local_loads:=[u for u in k.uops if u.op is UOps.LOAD and any([lb in u.src for lb in local_buf])]), 3, \
      f"should have generated 3 BufferOps.LOAD to the local buf but got {len(real_local_loads)}")
    self.assertEqual((real_local_stores[1].src[1].op, real_local_stores[1].src[1].arg), (UOps.CONST, 0))
    self.assertEqual((real_local_loads[1].src[1].op, real_local_loads[1].src[1].arg), (UOps.CONST, 0))
    x = Tensor.randn(3,27,32).realize()
    helper_linearizer_ast(ast, [x], wanna_output=[x.numpy().std(axis=2, ddof=0).reshape(-1)])

  @unittest.skip("AST has implicit movement ops")
  def test_multireduce_upcasting(self):
    # when upcasting multiple reductions, ensure ast_parse will create multiple uops even when using the result of past reductions
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float32, st=ShapeTracker(views=(View(shape=(8, 7), strides=(7, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=UnaryOps.NEG, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float32, st=ShapeTracker(views=(View(shape=(8, 7), strides=(7, 1), offset=0, mask=None, contiguous=True),),))),), arg=(1,)),), arg=None),)),), arg=(1,)),), arg=MemBuffer(idx=0, dtype=dtypes.float32, st=ShapeTracker(views=(View(shape=(8, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)))), # noqa: E501
    k = Kernel(ast)
    k.upcast()
    k.linearize()
    define_globals = [u for u in k.uops if u.op is UOps.DEFINE_GLOBAL]
    self.assertEqual(len([u for u in k.uops if u.op is UOps.LOAD and define_globals[1] in u.src]), 7)
    self.assertEqual(len([u for u in k.uops if u.op is UOps.ALU and u.arg is BinaryOps.ADD]), 25)
    opts = [[Opt(op=OptOps.UPCAST, axis=0, amt=2)], [Opt(op=OptOps.UPCAST, axis=0, amt=4)]]
    x = Tensor.randn(8,7).softmax().realize()
    helper_linearizer_ast(ast, [x], opts=opts, wanna_output=[(x.numpy() - x.numpy().sum(axis=1, keepdims=True)).sum(axis=1)])

  @unittest.skip("TODO: fix uops toposort")
  def test_multireduce_unroll(self):
    # unrolled multireduceops will cause an issue where and reduceop following another reduceop will need to bring the "unroll" back:
    # ex you unroll into four values, the four values sum, then you need to four operations on the sum for the next reduceop
    Tensor.manual_seed(0)
    x = Tensor.randn(3, 27, 12, dtype=dtypes.float).realize()
    first_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((3, 27, 1, 12)).expand((3, 27, 12, 12))))
    first_reduce = LazyOp(ReduceOps.SUM, (first_x,), (3,))
    mean = first_reduce * LazyOp(BufferOps.CONST, (), ConstBuffer(1/12, dtypes.float, ShapeTracker.from_shape(()).reshape((1, 1, 1, 1)).expand((3, 27, 12, 1)))) # noqa: E501
    second_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((3, 27, 12, 1))))
    squares = (second_x-mean)*(second_x-mean)
    squares_sum = LazyOp(ReduceOps.SUM, (squares,), (2,))
    variance = squares_sum * LazyOp(BufferOps.CONST, (), ConstBuffer(1/12, dtypes.float, ShapeTracker.from_shape(()).reshape((1, 1, 1, 1)).expand((3, 27, 1, 1)))) # noqa: E501
    store = LazyOp(BufferOps.STORE, (variance,), MemBuffer(0, dtypes.float, ShapeTracker.from_shape((3, 27, 1, 1))))
    wanna_output = x.numpy().var(axis=2, ddof=0).reshape((3,27,1,1))
    opts = [
      [Opt(op=OptOps.UNROLL, axis=0, amt=12)],
      [Opt(op=OptOps.UNROLL, axis=0, amt=6)],
      [Opt(op=OptOps.UNROLL, axis=0, amt=4)],
      [Opt(op=OptOps.UNROLL, axis=0, amt=3)],
      [Opt(op=OptOps.UNROLL, axis=0, amt=2)],
    ]
    helper_linearizer_ast((store,), [x], opts=opts, wanna_output=[wanna_output])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skip("AST has implicit movement ops")
  def test_multireduce_loop_scope(self):
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=UnaryOps.NEG, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))),), arg=(2,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.03125, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=None))), LazyOp(op=UnaryOps.RECIP, src=(LazyOp(op=UnaryOps.SQRT, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=UnaryOps.NEG, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))),), arg=(2,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.03125, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=None)), arg=None), LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=UnaryOps.NEG, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(864, 32, 1), offset=0, mask=None, contiguous=True),)))),), arg=(2,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.03125, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 32), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=None)), arg=None)), arg=None),), arg=(2,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.03125, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 1), strides=(27, 1, 0), offset=0, mask=None, contiguous=True),))))), arg=None),), arg=None),)),),),), arg=(2,)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3, 27, 1), strides=(27, 1, 0), offset=0, mask=None, contiguous=True),),))), # noqa: E501
    k = Kernel(ast)
    k.hand_coded_optimizations()
    k.linearize()
    def get_recursive_children(x:UOp): return set.union(set(x.src), *[get_recursive_children(v) for v in x.src])
    loop = None
    for u in k.uops:
      if u.op is UOps.RANGE: loop = u
      elif loop is None: continue
      elif u.op is UOps.ENDRANGE and loop in u.src: loop = None
      else: self.assertIn(loop, get_recursive_children(u), f"Any uop within a loop should depend on the loop: {u}")
    x = Tensor.randn(3, 27, 32).realize()
    helper_linearizer_ast(ast, [x], wanna_output= \
      [((x.numpy() - x.numpy().mean(axis=2, keepdims=True))/x.numpy().std(axis=2, keepdims=True, ddof=0)).sum(axis=2).reshape(-1)])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skip("TODO: fix uops toposort")
  def test_mean_std_multireduce(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(15, 25, 35, dtype=dtypes.float).realize()
    first_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((15, 25, 1, 35)).expand((15, 25, 35, 35))))
    first_reduce = LazyOp(ReduceOps.SUM, (first_x,), (3,))
    mean = first_reduce * LazyOp(BufferOps.CONST, (), ConstBuffer(1/35, dtypes.float, ShapeTracker.from_shape(()).reshape((1, 1, 1, 1)).expand((15, 25, 35, 1)))) # noqa: E501
    second_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((15, 25, 35, 1))))
    squares = (second_x-mean)*(second_x-mean)
    squares_sum = LazyOp(ReduceOps.SUM, (squares,), (2,))
    variance = squares_sum * LazyOp(BufferOps.CONST, (), ConstBuffer(1/35, dtypes.float, ShapeTracker.from_shape(()).reshape((1, 1, 1, 1)).expand((15, 25, 1, 1)))) # noqa: E501
    std = LazyOp(UnaryOps.SQRT, (variance,), None)
    store = LazyOp(BufferOps.STORE, (std,), MemBuffer(0, dtypes.float, ShapeTracker.from_shape((15, 25, 1, 1))))
    wanna_output = x.numpy().std(axis=2, ddof=0).reshape((15,25,1,1))
    helper_linearizer_ast((store,), [x], wanna_output=[wanna_output])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skip("TODO: fix uops toposort")
  def test_mean_std_multireduce_mid_dim(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(15, 25, 35, dtype=dtypes.float).realize()
    first_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((15, 1, 25, 35)).expand((15, 25, 25, 35))))
    first_reduce = LazyOp(ReduceOps.SUM, (first_x,), (2,))
    mean = first_reduce * LazyOp(BufferOps.CONST, (), ConstBuffer(0.04, dtypes.float, ShapeTracker.from_shape(()).reshape((1, 1, 1, 1)).expand((15, 25, 1, 35)))) # noqa: E501
    second_x = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, x.lazydata.st.reshape((15, 25, 1, 35))))
    squares = (second_x-mean)*(second_x-mean)
    squares_sum = LazyOp(ReduceOps.SUM, (squares,), (1,))
    variance = squares_sum * LazyOp(BufferOps.CONST, (), ConstBuffer(0.04, dtypes.float, ShapeTracker.from_shape(()).reshape((1, 1, 1, 1)).expand((15, 1, 1, 35)))) # noqa: E501
    std = LazyOp(UnaryOps.SQRT, (variance,), None)
    store = LazyOp(BufferOps.STORE, (std,), MemBuffer(0, dtypes.float, ShapeTracker.from_shape((15, 1, 1, 35))))
    wanna_output = x.numpy().std(axis=1, ddof=0).reshape((15,1,1,35))
    helper_linearizer_ast((store,), [x], wanna_output=[wanna_output])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.expectedFailure
  def test_mean_std_multireduce_multiout(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(15, 25, 35, dtype=dtypes.float).realize()
    first_x = LazyOp(BufferOps.LOAD, (), MemBuffer(2, dtypes.float, x.lazydata.st.reshape((15, 25, 1, 35)).expand((15, 25, 35, 35))))
    first_reduce = LazyOp(ReduceOps.SUM, (first_x,), (3,))
    mean = first_reduce * LazyOp(BufferOps.CONST, (), ConstBuffer(1/35, dtypes.float, ShapeTracker.from_shape(()).reshape((1, 1, 1, 1)).expand((15, 25, 35, 1)))) # noqa: E501
    second_x = LazyOp(BufferOps.LOAD, (), MemBuffer(2, dtypes.float, x.lazydata.st.reshape((15, 25, 35, 1))))
    squares = (second_x-mean)*(second_x-mean)
    squares_sum = LazyOp(ReduceOps.SUM, (squares,), (2,))
    variance = squares_sum * LazyOp(BufferOps.CONST, (), ConstBuffer(1/35, dtypes.float, ShapeTracker.from_shape(()).reshape((1, 1, 1, 1)).expand((15, 25, 1, 1)))) # noqa: E501
    std = LazyOp(UnaryOps.SQRT, (variance,), None)
    third_reduce = LazyOp(ReduceOps.SUM, (second_x,), (2,))
    mean_out = third_reduce * LazyOp(BufferOps.CONST, (), ConstBuffer(1/35, dtypes.float, ShapeTracker.from_shape(()).reshape((1, 1, 1, 1)).expand((15, 25, 1, 1)))) # noqa: E501
    store_mean = LazyOp(BufferOps.STORE, (mean_out,), MemBuffer(1, dtypes.float, ShapeTracker.from_shape((15,25,1,1))))
    store_std = LazyOp(BufferOps.STORE, (std,), MemBuffer(0, dtypes.float, ShapeTracker.from_shape((15, 25, 1, 1))))
    wanna_output = [x.numpy().std(axis=2, ddof=0).reshape(15,25,1,1), x.numpy().mean(axis=2).reshape(15,25,1,1)]
    lins = helper_linearizer_ast((store_std,store_mean), [x], wanna_output=wanna_output)

    for k in lins:
      assert len([u for u in k.uops if u.op is UOps.DEFINE_ACC]) == 2, "got more than two accs (didn't reuse the mean reduce)"

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skip("AST has implicit movement ops")
  def test_softmax_multireduce(self):
    x = Tensor.rand(4, 32).realize()
    x_ast = LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape((4,32))))
    max_x = LazyOp(op=ReduceOps.MAX, src=(x_ast,), arg=(1,))
    centered_x = LazyOp(op=BinaryOps.ADD, src=(x_ast, LazyOp(op=UnaryOps.NEG, src=(max_x,), arg=None)))
    exp_x = LazyOp(op=UnaryOps.EXP2, src=(centered_x,))
    sum_exp_x = LazyOp(op=ReduceOps.SUM, src=(exp_x,), arg=(1,))
    y = LazyOp(op=BinaryOps.MUL, src=(exp_x, LazyOp(op=UnaryOps.RECIP, src=(sum_exp_x,))))
    y_reduced = LazyOp(op=ReduceOps.SUM, src=(y,), arg=(1,))
    ast = LazyOp(op=BufferOps.STORE, src=(y_reduced,), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape((4,1))))
    expected = ((np_exp2:=np.exp2(x.numpy() - x.numpy().max(axis=-1, keepdims=True)))/np_exp2.sum(axis=-1, keepdims=True)).sum(axis=-1)
    helper_linearizer_ast((ast,), [x], wanna_output=[expected])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skip("AST has implicit movement ops")
  def test_softmax_multireduce_multiout(self):
    x = Tensor.rand(4, 32).realize()
    x_ast = LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker.from_shape((4,32))))
    max_x = LazyOp(op=ReduceOps.MAX, src=(x_ast,), arg=(1,))
    exp_x = LazyOp(op=UnaryOps.EXP2, src=(LazyOp(op=BinaryOps.ADD, src=(x_ast, LazyOp(op=UnaryOps.NEG, src=(max_x,), arg=None))),))
    sum_exp_x = LazyOp(op=ReduceOps.SUM, src=(exp_x,), arg=(1,))
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(exp_x, LazyOp(op=UnaryOps.RECIP, src=(sum_exp_x,)))),), arg=(1,)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape((4,1)))) # noqa: E501
    max_x_ast = LazyOp(op=BufferOps.STORE, src=(max_x,), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape((4,1))))
    sum_exp_x_ast = LazyOp(op=BufferOps.STORE, src=(sum_exp_x,), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker.from_shape((4,1))))
    expected = [
      ((np_exp2:=np.exp2(x.numpy()-(np_max_x:=x.numpy().max(axis=-1,keepdims=True))))/(sum_exp_x:=np_exp2.sum(axis=-1,keepdims=True))).sum(axis=-1,),
      np_max_x.reshape(-1), sum_exp_x.reshape(-1)
    ]
    helper_linearizer_ast((ast,max_x_ast,sum_exp_x_ast), [x], wanna_output=expected)

  def test_load_dedup(self):
    # for different leaves in the AST, the same loads may occur.

    a = Tensor.randn(4).realize()
    # these are of size 3 to avoid float4 coalesce
    r = a[:-1] + a[1:]

    k = Kernel(create_schedule([r.lazydata])[-1].ast)
    k.upcast()
    k.linearize()
    num_loads = len([uop for uop in k.uops if uop.op is UOps.LOAD])
    assert num_loads <= 4, "more load uops than needed"
    assert num_loads >= 4, "unexpected number of uops, maybe this test needs updating?"

  def test_load_cache_const_bufs(self):
    # make sure const buffers are differentiated from local and mem buffers
    ST, DT = ShapeTracker(views=(View(shape=((1,)), strides=(0, 0), offset=0, mask=None, contiguous=False),)), dtypes.int
    VAL = LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=2, dtype=DT, st=ST))

    # data1[0] + VAL
    a = LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=DT, st=ST)), VAL))
    # (literal const 1) + VAL
    b = LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1, dtype=DT, st=ST)), VAL))

    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=BinaryOps.ADD, src=(a,b)),), arg=MemBuffer(idx=0, dtype=DT, st=ST))
    lin = Kernel(ast)
    lin.linearize()

    assert len(lin.uops.uops) <= 7, "too many uops"
    a_bufs = [u.op for u in lin.uops.uops[-1].src[2].src]
    assert a_bufs == [UOps.LOAD, UOps.CONST]

  def test_upcast_cse(self):
    # when upcasting, within a subtree, there may be common expressions.

    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = a.expand([2]) + b.expand([2])

    k = Kernel(create_schedule([r.lazydata])[-1].ast)
    k.upcast()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.op is UOps.ALU])
    assert num_ops <= 1, "more alu uops than needed"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_reduce_upcast(self):
    x, w = Tensor.randn((1,1,3)).realize(), Tensor.randn((1,1,2)).realize()
    r = Tensor.conv2d(x,w,padding=1).relu()

    k = Kernel(create_schedule([r.lazydata])[-1].ast)
    k.upcast()
    k.upcast()
    k.linearize()
    accs = [u for u in k.uops if u.op is UOps.DEFINE_ACC]
    stores = [u for u in k.uops if u.op is UOps.STORE]
    assert len(accs) == 0  # it's removed now
    assert len(stores) == 1
    assert stores[0].src[-1].dtype == dtypes.float.vec(4)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_upcast_with_locals(self):
    x, y = Tensor.rand(1,128), Tensor.rand(128, 128)
    r = (x@y).relu()
    k = Kernel(create_schedule([r.lazydata])[-1].ast)
    k.hand_coded_optimizations()
    k.linearize()

    accs = [u for u in k.uops if u.op is UOps.DEFINE_ACC]
    stores = [u for u in k.uops if u.op is UOps.STORE]

    # the first store is to lds and can be upcasted
    assert accs[0].dtype == stores[0].src[-1].dtype == dtypes.float.vec(4)
    assert stores[0].src[0].op is UOps.DEFINE_LOCAL
    # the second store is to gds with no upcasts
    assert accs[1].dtype == stores[1].src[2].dtype == dtypes.float
    assert stores[1].src[0].op is UOps.DEFINE_GLOBAL

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skip("AST has implicit movement ops")
  def test_upcast_multireduce_nested_local_upcast(self):
    x, y, z, w = [Tensor.rand((1,128) if i % 2 == 0 else (1,128,128)).realize() for i in range(4)]
    st0 = ShapeTracker(views=(View(shape=(1, 128, 128), strides=(0, 0, 1), offset=0, mask=None, contiguous=False),))
    st1 = ShapeTracker(views=(View(shape=(1, 128, 128), strides=(0, 1, 128), offset=0, mask=None, contiguous=False),))
    ld0 = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, st0))
    ld1 = LazyOp(BufferOps.LOAD, (), MemBuffer(2, dtypes.float, st1))
    ld2 = LazyOp(BufferOps.LOAD, (), MemBuffer(3, dtypes.float, st0))
    ld3 = LazyOp(BufferOps.LOAD, (), MemBuffer(4, dtypes.float, st1))
    r0 = LazyOp(ReduceOps.SUM, (LazyOp(BinaryOps.MUL, (ld0, ld1)), ), (2,))
    r1 = LazyOp(ReduceOps.SUM, (LazyOp(BinaryOps.MUL, (ld2, ld3)), ), (2,))
    out_st = ShapeTracker(views=(View(shape=(1, 128, 1), strides=(0, 1, 0), offset=0, mask=None, contiguous=True),))
    ast = (LazyOp(BufferOps.STORE, (LazyOp(BinaryOps.ADD, (r0, r1)), ), MemBuffer(0, dtypes.float, out_st)),)
    helper_linearizer_ast(ast, [x, y, z, w])

  def test_zero_fold(self):
    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = Tensor.stack(a, b)

    k = Kernel(create_schedule([r.lazydata])[-1].ast)
    k.upcast()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.op is UOps.ALU])
    assert num_ops == 0, "more alu uops than needed"

  def test_sum_acc_dtype(self):
    for tensor_dtype, acc_dtype in (
      (dtypes.bool, dtypes.int), (dtypes.int16, dtypes.int), (dtypes.float16, dtypes.float), (dtypes.bfloat16, dtypes.float)):
      a = Tensor([1, 2, 3], dtype=tensor_dtype).sum()
      k = Kernel(create_schedule([a.lazydata])[-1].ast)
      k.linearize()
      local = [uop for uop in k.uops if uop.op is UOps.DEFINE_ACC]
      assert local[0].dtype == acc_dtype

  def test_arg_acc_dtype(self):
    def helper_arg_acc_dtype(c: Tensor, expected_dtype:DType):
      k = Kernel(create_schedule([c.lazydata])[-1].ast)
      k.linearize()
      local = [uop for uop in k.uops if uop.op is UOps.DEFINE_ACC]
      assert local[0].dtype == expected_dtype

    tests = (
      (dtypes.float16, None, dtypes.float),
      (dtypes.bfloat16, None, dtypes.float),
      (dtypes.float, None, dtypes.float),
      (dtypes.float16, dtypes.float16, dtypes.float16),
      (dtypes.bfloat16, dtypes.bfloat16, dtypes.bfloat16),
      (dtypes.float, dtypes.float16, dtypes.float16),
    )
    for tensor_dtype, acc_dtype, expected_dtype in tests:
      a, b = Tensor.rand(8, 8, dtype=tensor_dtype), Tensor.rand(8, 8, dtype=tensor_dtype)
      helper_arg_acc_dtype(a.sum(acc_dtype=acc_dtype), expected_dtype)
      helper_arg_acc_dtype(a.matmul(b, acc_dtype=acc_dtype), expected_dtype)
      helper_arg_acc_dtype(Tensor.einsum("ki,ij->kj", a, b, acc_dtype=acc_dtype), expected_dtype)
      d, w = Tensor.rand(4, 8, 8, 8, dtype=tensor_dtype), Tensor.rand(8, 8, 2, 2, dtype=tensor_dtype)
      helper_arg_acc_dtype(d.conv2d(w, acc_dtype=acc_dtype), expected_dtype)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if getenv("EMULATE_CUDA") and (tc.dtype_in == dtypes.bfloat16 or tc.dtype_out == dtypes.bfloat16): continue
      helper_tc_allclose(tc.dims[0], tc.dims[1], tc.dims[2], tc.dtype_in, tc.dtype_out, axis=0, tc_opt=0)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_padded(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if getenv("EMULATE_CUDA") and (tc.dtype_in == dtypes.bfloat16 or tc.dtype_out == dtypes.bfloat16): continue
      pad = 1

      # check that TC is triggered for TC_OPT=2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=True)

      # check that TC is not triggered for TC_OPT<2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=1, ensure_triggered=False)
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=0, ensure_triggered=False)

      # check excessive padding doesn't trigger padded TC in TC_OPT=2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]//4, tc.dims[1], tc.dims[2], tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)
      helper_tc_ensure_uops_and_opts_count(tc.dims[0], tc.dims[1]//4, tc.dims[2], tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)
      helper_tc_ensure_uops_and_opts_count(tc.dims[0], tc.dims[1], tc.dims[2]//4, tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)

      # check correctness
      helper_tc_allclose(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad, tc.dtype_in, tc.dtype_out, tc_opt=2)

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI is really slow here")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_multi_reduce(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if tc.dtype_in == dtypes.bfloat16 or tc.dtype_out == dtypes.bfloat16: continue
      # this will be a M=G16, N=G32, M=G16, M=G16, K=R16, K=R16, K=R16 with 9 choices of TC MNK axes
      golden_result = None
      for axis in range(9):
        a = Tensor.rand(16, 16, 29, 29, dtype=tc.dtype_in).realize()
        b = Tensor.rand(32, 16, 16, 16, dtype=tc.dtype_in).realize()
        c = a.conv2d(b, padding=1, acc_dtype=tc.dtype_out)
        realized_ast, real_bufs = helper_realized_ast(c)

        k = Kernel(realized_ast)
        k.apply_tensor_cores(1, axis=axis, tc_opt=2)
        k.linearize()
        assert len([uop for uop in k.uops if uop.op is UOps.WMMA]) > 0, "tensor core not triggered"
        assert len([x for x in k.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"

        prg = CompiledRunner(k.to_program())
        real_bufs[0].copyin(np.zeros((real_bufs[0].size, ), dtype=_to_np_dtype(real_bufs[0].dtype)).data) # Zero to check that all values are filled
        prg.exec(real_bufs)
        result = np.frombuffer(real_bufs[0].as_buffer(), _to_np_dtype(real_bufs[0].dtype))

        # ensure the results for each choice of axis matches
        if golden_result is None: golden_result = np.frombuffer(real_bufs[0].as_buffer(), _to_np_dtype(real_bufs[0].dtype))
        np.testing.assert_allclose(result, golden_result, atol=0.1, rtol=0.15)

      # check that get_kernel_actions produces all 9 options
      from tinygrad.engine.search import get_kernel_actions
      tc_actions = [k for i, k in get_kernel_actions(Kernel(realized_ast), False).items() if k.applied_opts[0].op == OptOps.TC]
      assert len(tc_actions) == 9, f"get_kernel_actions should contain 9 possible TC actions, only got {len(tc_actions)}"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_unroll_phi(self):
    tc = Device[Device.DEFAULT].renderer.tensor_cores[0]
    x, y = Tensor.rand(128, 128, dtype=tc.dtype_in), Tensor.rand(128, 128, dtype=tc.dtype_in)
    r = x.matmul(y, acc_dtype=tc.dtype_out)
    k = helper_linearizer_opt(r, [[Opt(OptOps.UNROLL, 0, 4)]], apply_tc=True, atol=3e-2, rtol=1e-3)[-1]
    for u in k.uops:
      if u.op is UOps.WMMA:
        assert u.src[-1].src[0].op != UOps.PHI

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_unroll_casted_phi(self):
    tc = [tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in != tc.dtype_out][0]
    x, y = Tensor.rand(128, 128, dtype=tc.dtype_in), Tensor.rand(128, 128, dtype=tc.dtype_in)
    r = x.matmul(y, acc_dtype=tc.dtype_out)
    k = helper_linearizer_opt(r, [[Opt(OptOps.UNROLL, 0, 4)]], apply_tc=True, atol=3e-2, rtol=1e-3)[-1]
    for u in k.uops:
      if u.op is UOps.WMMA:
        assert u.src[-1].dtype == dtypes.float.vec(prod(tc.thread_local_sizes[2]))
        assert u.src[-1].src[0].op != UOps.PHI

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_unroll_casted_phi_with_children(self):
    # all PHI children are outside the loop
    tc = [tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in != tc.dtype_out][0]
    x, y = Tensor.rand(128, 128, dtype=tc.dtype_in), Tensor.rand(128, 128, dtype=tc.dtype_in)
    r = x.matmul(y, acc_dtype=tc.dtype_out).relu()
    k = helper_linearizer_opt(r, [[Opt(OptOps.UNROLL, 0, 4)]], apply_tc=True, atol=3e-2, rtol=1e-3)[-1]
    for u in k.uops:
      if u.op is UOps.WMMA:
        assert u.src[-1].dtype == dtypes.float.vec(prod(tc.thread_local_sizes[2]))
        assert u.src[-1].src[0].op != UOps.PHI

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_simple_unroll_no_between_phi_dependencies(self):
    x, y = Tensor.rand(128, 128), Tensor.rand(128, 128)
    r = (x@y).relu()
    k = helper_linearizer_opt(r, [[Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4)]])[-1]
    # the uops graph is RANGE -> DEFINE_ACC -> 4x ALU -> 4x PHI -> ENDRANGE
    for u in k.uops:
      if u.op is UOps.PHI:
        assert u.src[1].op is UOps.ALU
      # children of PHI are placed after ENDRANGE
      if any(x.op is UOps.PHI for x in u.src):
        end_range = [i for i, x in enumerate(k.uops) if x.op is UOps.ENDRANGE][0]
        assert end_range < k.uops.uops.index(u)

  def test_grouped_dims(self):
    def _assert_grouped_dims(prefix, dims, max_sizes, reverse_dims, expected_sizes):
      # TODO: fix reverse_dims
      idxs = get_grouped_dims(prefix, dims, max_sizes)
      loop_idxs = dedup(flatten([[y for y in sorted(list(x.sparents)) if y.op is UOps.SPECIAL] for x in idxs]))
      sizes = [x.arg[2] for x in loop_idxs]
      assert len(idxs) == len(dims), f"expected idxs to have same length as dims {len(dims)}, got {len(idxs)}"
      assert len(loop_idxs) == min(len(sizes), len(dims)), f"expected idxs to have length {min(len(sizes), len(dims))}, got {len(loop_idxs)}"
      assert sizes == expected_sizes, f"expected sizes={expected_sizes}, got {sizes=}"
      # TODO: add these back after uop symbolic
      # for i in range(len(dims)):
      #   assert idxs[i].max+1 == dims[i], f"idxs[{i}] should have max {dims[i]-1}"
      # for i in range(len(loop_idxs)):
      #   assert loop_idxs[i].expr.startswith(prefix), f"loop_idxs[{i}] must start with {prefix}"
      #   assert loop_idxs[i].max+1 == sizes[i], f"loop_idxs[{i}] should have max {sizes[i]-1}"

    # no-op
    _assert_grouped_dims("gidx", (2,), (16,16,16), False, [2])
    _assert_grouped_dims("gidx", (2,3), (16,16,16), False, [2,3])

    # check reverse dims
    # _assert_grouped_dims("gidx", (2,3), (16,16,16), True, [3,2])
    _assert_grouped_dims("gidx", (2,3,4), (16,16,16), False, [2,3,4])

    # test splitting globals
    # _assert_grouped_dims("gidx", (64,3,4), (16,16,16), False, [16,12,4])
    # _assert_grouped_dims("gidx", (64,3,4), (16,4,16), False, [16,4,12])
    # _assert_grouped_dims("gidx", (64,3,4), (16,16,16), True, [12,16,4])
    # _assert_grouped_dims("gidx", (128,3,4), (16,4,256), False, [16,4,24])

    # collapse on onto the left most axis
    _assert_grouped_dims("gidx", (2,3,4,5), (16,16,16), False, [6,4,5])
    # _assert_grouped_dims("gidx", (2,3,4,5), (32,16,16), True, [20,3,2])
    # _assert_grouped_dims("gidx", (Variable("start_pos",1,2),3,4,5), (32,16,16), True, [20,3,Variable("start_pos",1,2)])

    # collapse on left-most available axis (the left most is too small)
    # _assert_grouped_dims("gidx", (2,3,4,5), (4,16,16), False, [2,12,5])
    # _assert_grouped_dims("gidx", (2,3,4,5), (16,16,16), True, [5,12,2])

    _assert_grouped_dims("gidx", (Variable("start_pos",1,2),3,4,5), (16,16,16), False, [Variable("start_pos",1,2)*3,4,5])

    # # dim too large and not factorable
    # with self.assertRaises(AssertionError):
    #   get_grouped_dims("gidx", 0, (23,), (16,16,16), False,)
    # with self.assertRaises(AssertionError):
    #   get_grouped_dims("gidx", 0, (128,3,4), (16,4,23), False,)

    # # too large for sizes
    # with self.assertRaises(AssertionError):
    #   get_grouped_dims("gidx", 0, (2,3,4,5,6), (16,16,16), False,)

    # # variable too large
    # with self.assertRaises(AssertionError):
    #   get_grouped_dims("gidx", 0, (Variable("start_pos",0,16),3,4), (16,16,16), False,)

  def test_div_collapse(self):
    def helper(t, msg, max_ops=0):
      sched = [si for si in create_schedule([t.lazydata]) if si.ast.op is MetaOps.KERNEL]
      assert len(sched) == 1

      lin = Kernel(sched[0].ast)
      assert sum(u.arg is UnaryOps.RECIP for u in lin.linearize().uops) == max_ops, msg

    a = Tensor.rand((4,4))
    b = Tensor.rand((4,4))
    d = Tensor.rand((4,4))

    c = (a*b)/b
    helper(c, "found UnaryOps.RECIP in (a*b)/b operation")

    c = a/a
    helper(c, "found UnaryOps.RECIP in (a/a) operation")

    c = (a/b)/d
    helper(c, "found multiple UnaryOps.RECIP in (a/b)/d operation", 1)

  def test_sum_collapse(self):
    t = Tensor([2]).reshape(1, 1).expand(256, 256).sum()
    sched = [si for si in create_schedule([t.lazydata]) if si.ast.op is MetaOps.KERNEL]
    assert len(sched) == 1
    lin = Kernel(sched[0].ast)
    assert not any(u.op is UOps.RANGE for u in lin.linearize().uops), "found loop in sum collapse"

  def test_assign_fold(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    m = Tensor.ones(4, 4).shrink(((1, 2), None)).pad(((1, 2), None))
    a.assign(a+m)
    a.realize()
    np.testing.assert_equal(a.flatten().numpy(), [1.,1.,1.,1.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,1.,1.])

  def test_where_fold(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.shrink(((1, 2), None)).pad(((1, 2), None))
    a.assign(b.where(2, a))
    sched = create_schedule([a.lazydata])
    assert len(sched) == 1
    sched_copy = sched[:]
    run_schedule(sched)
    np.testing.assert_equal(a.flatten().numpy(), [1.,1.,1.,1.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,1.,1.])
    lin = Kernel(sched_copy[-1].ast)
    lin.hand_coded_optimizations()
    lin.linearize()
    assert not any(u.arg == TernaryOps.WHERE for u in lin.uops), "found where where where should be folded"

  def test_phi_simplification(self):
    def helper(t, max_ops=0):
      k = helper_linearizer_opt(t)[-1]
      uops = list(k.linearize().uops)
      # ignore kernel optimized IF statements for now
      if if_op:=next((u for u in uops if u.op is UOps.IF), None):
        uops = uops[:uops.index(if_op)]
      assert len(set([u.op for u in uops if u.op in {UOps.RANGE, UOps.SPECIAL}])) == 1, "has either specials or ranges, not both"
      assert len([u for u in uops if u.op is UOps.PHI]) == 0, "PHI should have been simplified"
      # TODO: once uops track min/max this will be fixed
      #assert len([u for u in uops if u.arg is BinaryOps.MAX]) <= max_ops, "no unnecessary MAX ops"

    helper(Tensor.arange(5.5, (3.5*300), 3.5), max_ops=2)
    helper(Tensor.arange(-1, -100, -5), max_ops=2)
    # NOTE: both of these split the reduce (this just wasn't tracked before)
    #helper(Tensor.arange(-3.2, 6.7, 0.64), max_ops=2)
    #helper(Tensor.arange(256), max_ops=2)
    helper(Tensor.arange(255), max_ops=2)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_grouped_store_phis(self):
    """
    float4 acc0 = float4(0.0,0.0,0.0,0.0);
    {
      acc0 = // ...
    }
    *((device float4*)(data0+alu2)) = float4(acc0.x,acc0.y,acc0.z,acc0.w);
    simplifies to:
    *((device float4*)(data0+alu2)) = acc0;
    """
    x, y = Tensor.randn(64,64), Tensor.randn(64,64)
    out = x.matmul(y)
    k = helper_linearizer_opt(out)[-1]
    # check that the float4 cast collapses
    store_vals = [u.src[-1] for u in k.uops if u.op is UOps.STORE]
    for val in store_vals:
      assert val.dtype == dtypes.float.vec(4) and val.op is not UOps.VECTORIZE

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_grouped_store_values(self):
    x = Tensor.randn((4,3,6,6)).realize()
    out = x.flip((0,1)).contiguous()
    k = helper_linearizer_opt(out)[-1]
    store_val = [u.src[-1] for u in k.uops if u.op is UOps.STORE][0]
    assert store_val.dtype == dtypes.float.vec(4) and store_val.op is not UOps.VECTORIZE

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_grouped_store_locals_and_globals(self):
    x, y = Tensor.rand(128, 128), Tensor.rand(128, 128)
    out = x@y
    opt = [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8),
            Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 2)] # upcast accs in both reduces
    k = helper_linearizer_opt(out, opts=[opt])[-1]
    def get_recursive(uop): return set.union(set(uop.src), [uop], *[get_recursive(v) for v in uop.src])
    local_stores = [u for u in k.uops if u.op is UOps.STORE and any(x.op is UOps.DEFINE_LOCAL for x in get_recursive(u.src[0]))]
    global_stores = [u for u in k.uops if u.op is UOps.STORE and any(x.op is UOps.DEFINE_GLOBAL for x in get_recursive(u.src[0]))]
    barrier = [u for u in k.uops if u.op is UOps.BARRIER][0]
    # check that the float4 cast collapses for all stores
    for store in local_stores+global_stores:
      assert store.src[2].dtype == dtypes.float.vec(2) and store.src[2].op is not UOps.VECTORIZE
    # # check the children's vins
    # TODO: src ALU are not the same, should it?
    # assert barrier.src == tuple(local_stores)
    assert len([u for u in k.uops if u.op is UOps.IF and u.src[-1] == barrier]) == 1

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_grouped_store_local_only(self):
    x, y = Tensor.rand(1,128), Tensor.rand(128, 128)
    r = (x@y).relu()
    k = helper_linearizer_opt(r)[-1]
    stores = [u for u in k.uops if u.op is UOps.STORE]

    # the float4 value stores directly in lds and we skip upcast
    assert stores[0].src[-1].dtype == dtypes.float.vec(4)
    assert stores[0].src[-1].op is not UOps.VECTORIZE

    # the global store doesn't change
    assert stores[1].src[2].dtype == dtypes.float

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_skip_unmatching_upcasts(self):
    Tensor.manual_seed(0)
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(240, 40, 1, 1), strides=(1, 240, 0, 0), offset=0, mask=None, contiguous=False),)))),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(240, 40, 1, 1), strides=(40, 1, 0, 0), offset=0, mask=None, contiguous=True),)))), # noqa: E501
    opt = [
        Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=0, amt=16),
        Opt(op=OptOps.LOCAL, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=2)
    ]
    k = helper_linearizer_ast(ast, [Tensor.randn(240*40).realize()], opts=[opt])[-1]
    out = [u for u in k.uops if u.op is UOps.STORE][0]
    assert out.src[-1].op is UOps.VECTORIZE and out.src[-1].dtype == dtypes.float.vec(4)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  @unittest.expectedFailure  # this will require compaction of BinaryOps.ADD
  def test_skip_unmatching_upcasts_with_gep(self):
    Tensor.manual_seed(0)
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(8, 32, 1, 1), strides=(1, 8, 0, 0), offset=0, mask=None, contiguous=False),)))),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(8, 32, 1, 1), strides=(32, 1, 0, 0), offset=0, mask=None, contiguous=True),)))), # noqa: E501
    opt = [Opt(op=OptOps.LOCAL, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=2, amt=2), Opt(op=OptOps.LOCAL, axis=1, amt=8),
            Opt(op=OptOps.UPCAST, axis=1, amt=0), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=0, amt=8),
            Opt(op=OptOps.UPCAST, axis=1, amt=0), Opt(op=OptOps.UPCAST, axis=0, amt=2)]
    k = helper_linearizer_ast(ast, [Tensor.randn(8*32).realize()], opts=[opt])[-1]
    out = [u for u in k.uops if u.op is UOps.STORE][0]
    assert out.src[-1].op is UOps.VECTORIZE and out.src[-1].dtype == dtypes.float.vec(2)

@unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need backends that support float4")
class TestFloat4(unittest.TestCase):
  @staticmethod
  def count_float4(k):
    return (len([uop for uop in k.uops if uop.op is UOps.LOAD and uop.dtype == dtypes.float.vec(4)]),
            len([uop for uop in k.uops if uop.op is UOps.STORE and len(uop.src) == 3 and uop.src[2].dtype == dtypes.float.vec(4)]))

  # TODO: express opts below as auto opts

  def test_float4_basic(self):
    a = Tensor.rand(2, 8).realize()
    b = Tensor.rand(2, 8).realize()
    c = a + b

    s = create_schedule([c.lazydata])[0]
    k = Kernel(s.ast)
    k.hand_coded_optimizations()
    k.linearize()

    assert TestFloat4.count_float4(k) == (2, 1)

  def test_float4_multidim(self):
    a = Tensor.rand(2, 8).realize()
    b = Tensor.rand(2, 8).realize()
    c = a + b

    s = create_schedule([c.lazydata])[0]
    k = Kernel(s.ast)
    k.shift_to(0, 4)  # float4 dimension
    k.shift_to(0, 2, insert_before=k.shape_len-1)
    k.upcast()
    k.upcast()
    k.local_dims += 1
    k.linearize()

    assert TestFloat4.count_float4(k) == (4, 2)

  def test_float4_unaligned_load(self):
    a = Tensor.rand(9).realize().shrink(((1, 9),))
    b = Tensor.rand(9).realize().shrink(((1, 9),))
    c = a + b

    s = create_schedule([c.lazydata])[0]
    k = Kernel(s.ast)
    k.hand_coded_optimizations()  # implicit trigger float4 dim
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 1)

  def test_float4_multidim_unaligned_load(self):
    a = Tensor.rand(2, 9).realize().shrink(((0, 2), (1, 9),))
    b = Tensor.rand(2, 9).realize().shrink(((0, 2), (1, 9),))
    c = a + b

    s = create_schedule([c.lazydata])[0]
    k = Kernel(s.ast)
    k.shift_to(len(k.full_unupcasted_shape)-1, 4)  # manual trigger float4 dim
    k.upcast()
    k.shift_to(len(k.full_unupcasted_shape)-1, 2, insert_before=k.shape_len-1)
    k.upcast()
    k.local_dims += 1
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 2)

  def test_float4_sometimes_unaligned(self):
    a = Tensor.rand(1, 1, 8).realize()
    b = Tensor.rand(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # only the first and last conv dot products are aligned in a, and b is never aligned, so no
    # float4 should be emitted (the reduce axis of size 4 is the float4 axis here)

    s = create_schedule([c.lazydata])[0]
    k = Kernel(s.ast)
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 0)

  def test_float4_multidim_sometimes_unaligned(self):
    a = Tensor.rand(1, 1, 7).realize()
    b = Tensor.rand(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # the first conv dot product is aligned in a. If we upcast the output and reduce
    # dimension, then we could do float4 for only that one set of loads, but we currently
    # don't.

    s = create_schedule([c.lazydata])[0]
    k = Kernel(s.ast)
    k.upcast()
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 1)

  def test_float4_noncontiguous(self):
    a = Tensor.rand(4, 2).realize()
    b = Tensor.rand(4, 2).realize()
    c = a + b

    # we will upcast the top axis of sz 4. they should not be coalesced into float4,
    # since the top axis is not contiguous.

    s = create_schedule([c.lazydata])[0]
    k = Kernel(s.ast)
    k.shift_to(0, 4, top=True)  # top axes are float4 axes
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 0)

  def test_float4_expand(self):
    a = Tensor.rand(9).realize().shrink(((1, 9),))
    b = Tensor.rand(2).realize().reshape((2, 1)).expand((2,4)).reshape((8,))
    c = a + b

    # we will upcast the top axis of sz 4. they should not be coalesced into float4,
    # since the top axis is not contiguous.

    s = create_schedule([c.lazydata])[0]
    k = Kernel(s.ast)
    k.shift_to(0, 4)  # float4 axis
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 1)

  def test_float4_heterogeneous(self):
    a = Tensor.rand(8).realize()
    b = Tensor.rand(9).realize().shrink(((1, 9),))
    c = a + b

    # should float4 b but not a

    s = create_schedule([c.lazydata])[0]
    k = Kernel(s.ast)
    k.shift_to(0, 4)  # float4 axis
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (1, 1)

class TestHandCodedOpts(unittest.TestCase):
  def test_masked_upcast(self):
    layer_1 = Tensor.cat(*[Tensor.rand(5) for _ in range(4)])
    layer_2 = Tensor.cat(layer_1.unsqueeze(0), Tensor.rand(6, 20))

    s = create_schedule([layer_2.lazydata])[-1]
    k = Kernel(s.ast)
    k.hand_coded_optimizations()
    assert len(k.bufs) == 6  # make sure all ops are done in one kernel
    # masked upcast should upcast masked axis of size 7
    # masked upcast should not upcast large (20) last axis
    # float4/other hcopt shouldn't upcast last axis, since we already have 7 upcast, and the last axis is not very contiguous
    assert k.upcasted == 1 and k.full_shape[-1] == 7

  def test_masked_upcast_wino(self):
    monster = Tensor.stack(*[Tensor.stack(*[Tensor.rand(16) for _ in range(6)]) for _ in range(6)])

    s = create_schedule([monster.lazydata])[-1]
    k = Kernel(s.ast)
    k.hand_coded_optimizations()
    assert len(k.bufs) == 37  # make sure all ops are done in one kernel
    # should upcast the two Tensor.stacks
    assert k.upcasted >= 2 and k.full_shape[k.shape_len-k.upcasted:k.shape_len].count(6) == 2

  def test_masked_upcast_wino_full(self):
    with Context(WINO=1):
      x,w = Tensor.rand(1,4,8,8, requires_grad=True).realize(), Tensor.rand(4,4,3,3, requires_grad=True).realize()
      out = Tensor.conv2d(x,w, padding=1)
      upcasts = []
      wino_schedule = create_schedule([out.lazydata])
      # collect upcasts of tile transform kernels
      for i, si in enumerate(wino_schedule):
        k = Kernel(si.ast)
        k.hand_coded_optimizations()
        if k.reduceop is not None: continue  # not a tile transform kernel (there is a gemm reduce kernel)
        if len(k.bufs) < 36: continue  # not a tile transform kernel (there's a permute kernel at the end)
        upcasts.append(tuple(k.full_shape[k.shape_len - k.upcasted:k.shape_len]))
      assert len(upcasts) == 3  # 3 transformation matrices
      assert len(wino_schedule) <= 4  # 4 kernels
      # this test case's inputs are too small, so one of the 4-stacks became a local, which is fine i guess
      assert upcasts.count((6, 6)) == 2 #and upcasts.count((4, 4)) == 1

      out.mean().backward()
      backward_schedule = create_schedule([x.grad.lazydata, w.grad.lazydata])
      for si in backward_schedule:
        k = Kernel(si.ast)
        k.hand_coded_optimizations()
        k.linearize()
        if len(k.bufs) < 20: continue  # not a tile transform kernel
        # heuristic number to make sure that at least some upcasts but not too many upcasts are being done
        assert 6 <= prod(k.full_shape[k.shape_len - k.upcasted:k.shape_len]) <= 216
      assert len(backward_schedule) <= 13  # just the current number, but it could be better

  def test_masked_upcast_many(self):
    layer_1 = Tensor.cat(Tensor.rand(3, 4), Tensor.rand(4, 4))
    layer_2 = Tensor.cat(layer_1.unsqueeze(0), Tensor.rand(6, 7, 4))
    layer_3 = Tensor.cat(layer_2.unsqueeze(0), Tensor.rand(6, 7, 7, 4))

    k = helper_linearizer_opt(layer_3)[-1]
    assert len(k.bufs) == 5  # make sure all ops are done in one kernel
    # check that we don't do too many upcasts
    assert prod(k.full_shape[k.shape_len-k.upcasted:k.shape_len]) <= 49

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_matvec(self):
    N = 128
    a = Tensor.rand(1, N).realize()
    b = Tensor.rand(N, N).realize()
    c = a @ b

    k = helper_linearizer_opt(c)[-1]

    assert k.group_for_reduces == 1
    assert k.local_dims == 1
    assert k.upcasted == 1

def helper_linearizer_ast(ast:Union[Tuple[LazyOp, ...], LazyOp], inputs:List[Tensor], *args, **kwargs):
  if not isinstance(ast, LazyOp): ast = LazyOp(MetaOps.KERNEL, ast)
  inbufs = [x.lazydata.buffer for x in inputs]
  outbufs = [Buffer(inbufs[-1].device if inbufs else Device.DEFAULT, out.arg.st.size, out.arg.dtype).allocate() for out in ast.src]
  return _helper_linearizer_opt_ast(ast, outbufs+inbufs, *args, **kwargs)

def helper_linearizer_opt(r:Union[Tensor, List[Tensor]], *args, **kwargs):
  realized_ast, real_bufs = helper_realized_ast(r)
  return _helper_linearizer_opt_ast(realized_ast, real_bufs, *args, **kwargs)

def _helper_linearizer_opt_ast(realized_ast:LazyOp, real_bufs:List[Buffer], opts=[],
                               apply_tc=False, atol=1e-4, rtol=1e-4, color_sizes=[], wanna_output=[]) -> List[Kernel]:
  lins: List[Kernel] = []
  outbufs = [(real_bufs[i], lop.arg.st.shape) for i,lop in enumerate(realized_ast.src)]

  def get_prg(k:Kernel): return CompiledRunner(replace(k.to_program(), dname=Device.DEFAULT))

  def check_opt(opts, create_k, expected_color_size):
    k = create_k()
    lins.append(k)
    if apply_tc:
      assert k.apply_tensor_cores(1, extra_opts=opts), "no tensor core triggered"
    else:
      for opt in opts:
        k.apply_opt(opt)
    if expected_color_size is not None:
      assert (cs:=list(zip(k.colors(), k.full_shape))) == expected_color_size, f"expected={expected_color_size} got={cs}"
    prg = get_prg(k)
    for buf,_ in outbufs: buf.copyin(np.zeros((buf.size, ), dtype=_to_np_dtype(buf.dtype)).data) # Zero to check that all values are filled
    prg.exec(real_bufs)

    for i, (buf,shape) in enumerate(outbufs):
      np.testing.assert_allclose(np.frombuffer(buf.as_buffer(), _to_np_dtype(buf.dtype)).reshape(shape), wanna_output[i], atol=atol, rtol=rtol)

  # Get baseline if it is not provided, which is not optimized at all.
  k = Kernel(realized_ast)
  lins.append(k)
  prg = get_prg(k)
  prg.exec(real_bufs)
  if len(wanna_output) == 0: wanna_output = [np.frombuffer(buf.as_buffer(), _to_np_dtype(buf.dtype)).reshape(shape).copy() for buf,shape in outbufs]
  else:
    for i, (buf,shape) in enumerate(outbufs):
      np.testing.assert_allclose(np.frombuffer(buf.as_buffer(), _to_np_dtype(buf.dtype)).reshape(shape), wanna_output[i], atol=atol, rtol=rtol)

  # Check correctness of handcoded optimiztions.
  k = Kernel(realized_ast)
  lins.append(k)
  k.hand_coded_optimizations()
  prg = get_prg(k)
  for buf,_ in outbufs: buf.copyin(np.zeros((buf.size, ), dtype=_to_np_dtype(buf.dtype)).data) # Zero to check that all values are filled
  prg.exec(real_bufs)
  for i, (buf,shape) in enumerate(outbufs):
    np.testing.assert_allclose(np.frombuffer(buf.as_buffer(), _to_np_dtype(buf.dtype)).reshape(shape), wanna_output[i], atol=atol, rtol=rtol)
  for i, x in enumerate(opts): # Check custom transformations if any.
    check_opt(x, lambda: Kernel(realized_ast), color_sizes[i] if i < len(color_sizes) else None)
  return lins

# creates a back-to-back multi reduce AST by merging r0 and r1.
# TODO: delete once we can schedule multi reduce
def _temp_create_multireduce_ast(r0:Tensor, r1:Tensor, replace_idxs:Dict[int,Tensor]={}, \
                                 merge=lambda r0,r1: LazyOp(BinaryOps.ADD, (r0, r1))) -> Tuple[LazyOp, ...]:
  assert len(s0:=r0.schedule()) == 1 and len(s1:=r1.schedule()) == 1, "inputs should be realized"
  assert all({idx:replace_idxs[idx] is r0 or replace_idxs[idx] is r1 for idx in replace_idxs}.values()), "replace idxs should be in {{r0, r1}}"
  op0, op1 = s0[0].ast.src[0].src[0], s1[0].ast.src[0].src[0]
  _replace_idxs = {idx:(op0 if replace_idxs[idx] is r0 else op1) for idx in replace_idxs}
  def _deep_replace(op:LazyOp, offset=0):
    if op.op is BufferOps.LOAD:
      if op.arg.idx+offset in _replace_idxs: return _replace_idxs[op.arg.idx+offset]
      else: arg = MemBuffer(op.arg.idx+offset, op.arg.dtype, op.arg.st)
    else: arg = op.arg
    return LazyOp(op.op, tuple(_deep_replace(x, offset) for x in op.src), arg)
  # limitation: r0 and r1 cannot share inputs.
  op0 = _deep_replace(op0, 0)
  op0_loads = len([x for x in op0.lazyops if x.op is BufferOps.LOAD])
  out = merge(op0, _deep_replace(op1, op0_loads))
  # limitation: only tests single output
  op = LazyOp(BufferOps.STORE, (out, ), MemBuffer(0, s0[-1].ast.src[-1].arg.dtype, s0[-1].ast.src[-1].arg.st))
  if DEBUG >= 3: print_tree(op)
  return op,

def check_fused_tc_opt(tc:TensorCore, r0:Tensor, r1:Tensor, inputs:List[Tensor]):
  ast = _temp_create_multireduce_ast(r0, r1)
  (atol, rtol) = ((0.25, 0.01) if tc.dtype_out == dtypes.half else (3e-2, 1e-3)) if tc.dtype_in == dtypes.half else (1e-4, 1e-4)
  helper_linearizer_ast(ast, inputs, [
    [],
    [Opt(OptOps.UPCAST, 0, 4)],
    [Opt(OptOps.UPCAST, 1, 4)],
    [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)], # check upcasts
    [Opt(OptOps.UNROLL, 0, 2)], # check unroll
    [Opt(OptOps.UNROLL, 0, 0)], # check full unroll of reduce with locals
    [Opt(OptOps.LOCAL, 0, 4)], # check local
    [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 2)], # check combo of unroll and local
    [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 2)],
    [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4)],
    [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.LOCAL, 0, 2)],
    [Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4)], # check permutations
    [Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 0, 4)],
    [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 4)],
    [Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 4)],
    [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 0, 4)],
    # [Opt(OptOps.GROUP, 0, 2)] # doesn't work because group_for_reduce dims become early locals (conflicting with TC)
  ], apply_tc=True, atol=atol, rtol=rtol)

class TestKernelOpts(unittest.TestCase):
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_local_and_grouped_reduce(self):
    N = 128
    Tensor.manual_seed(1882)
    a = Tensor.rand(4, 4, N, N)
    b = Tensor.rand(4, 4, N)
    r = (b.sqrt() + ((a+1).sum(axis=3).exp()))
    helper_linearizer_opt(r, [
      [Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 8)],
      [Opt(OptOps.LOCAL, 0, 16)], # Checking how it works with locals
      [Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 0, 64)], # Checking how it works with grouped reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.GROUPTOP, 0, 16)],
      [Opt(OptOps.LOCAL, 0, 32), Opt(OptOps.GROUPTOP, 0, 2)],
      # Checking how it works with locals + grouped reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 64)],
      # Checking how it works with locals + grouped reduce + upcasts
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.UPCAST, 0, 8), Opt(OptOps.UNROLL, 1, 4)],
      # many local + many group
      [Opt(OptOps.GROUP, 0, 2)] * 4,
      [Opt(OptOps.LOCAL, 0, 2)] * 4,
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUP, 0, 2)] * 4,
    ])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skip("parallel reduce")
  def test_local_and_grouped_reduce_multireduce(self):
    N = 128
    Tensor.manual_seed(1882)
    a = Tensor.rand(4, 4, N, N).realize()
    b = Tensor.rand(4, 4, N).realize()
    # TODO: this isn't the best AST, it's always math.inf
    r0 = (b.sqrt() + ((a+1).sum(axis=3).exp()))
    c = Tensor.rand(4, 4, N, N).realize()
    d = Tensor.rand(4, 4, N).realize()
    r1 = (d.sqrt() + ((c+1).sum(axis=3).exp()))
    ast = _temp_create_multireduce_ast(r0, r1)
    helper_linearizer_ast(ast, [b, a, d, c], [
      [Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 8)],
      [Opt(OptOps.LOCAL, 0, 16)], # Checking how it works with locals
      [Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 0, 64)], # Checking how it works with grouped reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.GROUPTOP, 0, 16)],
      [Opt(OptOps.LOCAL, 0, 32), Opt(OptOps.GROUPTOP, 0, 2)],
      # Checking how it works with locals + grouped reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 64)],
      # Checking how it works with locals + grouped reduce + upcasts
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.UPCAST, 0, 8), Opt(OptOps.UNROLL, 1, 4)],
    ])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skip("AST has implicit movement ops")
  def test_atomic_store_multireduce(self):
    # reducops will need to use the local buffer to load the result of a local reduce into every thread, barriers are needed on both sides
    # of the load to ensure 1) the correct value is in the local buffer and 2) the value isn't overwritten by the next reduceop
    N = 512
    Tensor.manual_seed(1882)
    a,b = Tensor.rand(4,4,N).realize(), Tensor.rand(4,4,N).realize()
    r0,r1 = a.sum(-1), b.sum(-1)
    ast = _temp_create_multireduce_ast(r0, r1)
    lins = helper_linearizer_ast(ast, [a,b], [[Opt(OptOps.GROUP, 0, 2)]])

    # sequential
    a,b = Tensor.rand(4,4,N).realize(), Tensor.rand(4,4,N).realize()
    dummy = Tensor.rand(4,4,1).realize()
    r0,r1 = (a-dummy).sum(-1), b.sum(-1)
    ast = _temp_create_multireduce_ast(r0, r1, replace_idxs={2:r1}, merge=lambda r0,_: r0)
    lins += helper_linearizer_ast(ast, [a], [[Opt(OptOps.GROUP, 0, 2)]])

    for k in lins:
      seen_bar = False
      for u in k.uops:
        if u.op is UOps.BARRIER:
          assert not seen_bar, "redudant barrier"
          seen_bar = True
        elif (u.op is UOps.LOAD or u.op is UOps.STORE): seen_bar = False

  @unittest.skip("TODO: broken")
  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_atomic_store_unrolled_multireduce(self):
    # unrolled local dim - causes stores for local reductions to pool at the top of the kernel, overwriting eachother
    Tensor.manual_seed(1882)
    a,b = Tensor.rand(4,).realize(), Tensor.rand(4,).realize()
    r0,r1 = a.sum(), b.sum()
    ast = _temp_create_multireduce_ast(r0, r1)
    lins = helper_linearizer_ast(ast, [a,b], [
      [Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.GROUP, 0, 2)]
    ])

    for k in lins:
      seen_bar = False
      for u in k.uops:
        if u.op is UOps.BARRIER:
          assert not seen_bar, "redudant barrier"
          seen_bar = True
        elif (u.op is UOps.LOAD or u.op is UOps.STORE): seen_bar = False

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skip("AST has implicit movement ops")
  def test_atomic_store_nested_range_multireduce(self):
    # nested ranges
    Tensor.manual_seed(1882)
    a,b = Tensor.rand(6, ).realize(), Tensor.rand(6, ).realize()
    r0,r1 = a.reshape(6, 1).expand(6, 3).sum(), b.reshape(6, 1).expand(6, 3).sum()
    ast = _temp_create_multireduce_ast(r0, r1)
    lins = helper_linearizer_ast(ast, [a,b], [
      [Opt(OptOps.GROUP, 0, 2)],[Opt(OptOps.GROUP, 1, 3)],
      [Opt(OptOps.GROUP, 1, 3), Opt(OptOps.GROUP, 0, 2)],
      [Opt(OptOps.UNROLL, 0, 2)],[Opt(OptOps.UNROLL, 1, 3)],
      [Opt(OptOps.GROUP, 0, 2), Opt(OptOps.UNROLL, 0, 2)],
      [Opt(OptOps.GROUP, 1, 3), Opt(OptOps.UNROLL, 1, 3)],
    ])

    for k in lins:
      seen_bar = False
      for u in k.uops:
        if u.op is UOps.BARRIER:
          assert not seen_bar, "redudant barrier"
          seen_bar = True
        elif (u.op is UOps.LOAD or u.op is UOps.STORE): seen_bar = False

  def test_upcasts(self):
    N = 16
    Tensor.manual_seed(1772)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4)],
      [Opt(OptOps.UPCAST, 0, 8)], # Checking how it works with upcasts
    ])

  def test_full_upcast(self):
    Tensor.manual_seed(1772)
    a = Tensor.rand(4)
    b = Tensor.rand(4)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 4)], # Checking how it works with upcasts
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_matmul(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)], # Checking how it works with upcasts
      [Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 1, 32)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 32)],
      [Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.LOCAL, 1, 8)], # Checking how it works with locals
      [Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 0, 32), Opt(OptOps.UNROLL, 0, 4)], # Checking how it works with grouped_reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.GROUPTOP, 0, 4)], # Checking how it works with local+grouped_reduce
      # Checking all together
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4),
       Opt(OptOps.UPCAST, 1, 2)],
      # Full global upcast + local
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 8)],
    ])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skip("AST has implicit movement ops")
  def test_matmul_multireduce(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N).realize()
    b = Tensor.rand(N, N).realize()
    r0 = a@b
    c = Tensor.rand(N, N).realize()
    d = Tensor.rand(N, N).realize()
    r1 = c@d
    ast = _temp_create_multireduce_ast(r0, r1)
    helper_linearizer_ast(ast, [a, b, c, d], [
      [Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)], # Checking how it works with upcasts
      [Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 1, 32)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 32)],
      [Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.LOCAL, 1, 8)], # Checking how it works with locals
      [Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 0, 32), Opt(OptOps.UNROLL, 0, 4)], # Checking how it works with grouped_reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.GROUPTOP, 0, 4)], # Checking how it works with local+grouped_reduce
      # Checking all together
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4),
       Opt(OptOps.UPCAST, 1, 2)],
      # Full global upcast + local
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 8)],
    ], wanna_output=[(a.numpy()@b.numpy()+c.numpy()@d.numpy()).flatten()])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_double_reduce(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(8, N, 8, N)
    r = a.sum(axis=(1,3))
    helper_linearizer_opt(r, [
      # openCL / GPU=1 is 256 max threads
      [Opt(OptOps.GROUPTOP, 0, 2)], [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 1, 2)], [Opt(OptOps.GROUPTOP, 1, 32)], # Checking how it works with 1 grouped_reduce.
      [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 2)],
      [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2)],
      [Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 64)], # Checking how it works with 2 grouped_reduces.
      [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2), Opt(OptOps.UNROLL, 0, 4)],
      [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 2, 4)], # Checking how it works with 2 grouped_reduces + upcasts.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4)],
      # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 1, 4)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2),
       Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UNROLL, 1, 4)], # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2),
       Opt(OptOps.UPCAST, 0, 2)], # No globals
    ])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skip("AST has implicit movement ops")
  def test_double_reduce_multireduce(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(8, N, 8, N).realize()
    r0 = a.sum(axis=(1,3))
    b = Tensor.rand(8, N, 8, N).realize()
    r1 = b.sum(axis=(1,3))
    ast = _temp_create_multireduce_ast(r0, r1)
    helper_linearizer_ast(ast, [a, b], [
      # openCL / GPU=1 is 256 max threads
      [Opt(OptOps.GROUPTOP, 0, 2)], [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 1, 2)], [Opt(OptOps.GROUPTOP, 1, 32)], # Checking how it works with 1 grouped_reduce.
      [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 2)],
      [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2)],
      [Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 64)], # Checking how it works with 2 grouped_reduces.
      [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2), Opt(OptOps.UNROLL, 0, 4)],
      [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 2, 4)], # Checking how it works with 2 grouped_reduces + upcasts.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4)],
      # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 1, 4)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2),
       Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UNROLL, 1, 4)], # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2),
       Opt(OptOps.UPCAST, 0, 2)], # No globals
    ], wanna_output=[(a.numpy().sum(axis=(1, 3))+b.numpy().sum(axis=(1, 3))).flatten()])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_invalid_tensor_core_extra_opts(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    realized_ast, _ = helper_realized_ast(a@b)
    invalid_opts = [
      [Opt(OptOps.LOCAL, 2, 2)],
      [Opt(OptOps.UPCAST, 2, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 2, 2)],
    ]
    for x in invalid_opts:
      k = Kernel(realized_ast)
      with self.assertRaises(AssertionError):
        assert k.apply_tensor_cores(use_tensor_cores=1, extra_opts=x), "no valid tensor core" # for METAL in runners

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_buf_index_not_found_tensor_core(self):
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=BinaryOps.CMPNE, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(1243, 256), strides=(0, 1), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(1243, 256), strides=(1, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=dtypes.float), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1243, 256), strides=(1, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=(0,)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 256), strides=(0, 1), offset=0, mask=None, contiguous=True),))))  # noqa: E501
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    with self.assertRaises(KernelOptError):
      k.apply_opt(Opt(OptOps.TC, 0, 1))

  @unittest.skip("parallel tensor cores")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_invalid_fused_tensor_core(self):
    Tensor.manual_seed(1552)
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if tc.dtype_in == dtypes.bfloat16: continue
      M, N, K = 12, 8, 30
      a, b = Tensor.rand(M, K, dtype=tc.dtype_in).realize(), Tensor.rand(K, N, dtype=tc.dtype_in).realize()
      r0 = a.matmul(b, acc_dtype=tc.dtype_out)
      M, N, K = 16, 8, 33
      c, d = Tensor.rand(M, K, dtype=tc.dtype_in).realize(), Tensor.rand(K, N, dtype=tc.dtype_in).realize()
      r1 = c.matmul(d, acc_dtype=tc.dtype_out)
      ast = _temp_create_multireduce_ast(r0, r1)
      lin = Kernel(ast)
      lin.apply_opt(Opt(op=OptOps.TC, axis=0, amt=2))
      lin.linearize()
      result = compare_linearizer(lin)
      assert result[0] == "COMPARE_ERROR"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_core_opts(self):
    N = 128
    Tensor.manual_seed(1552)
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      # bf16 buffer returns float32 numpy outputs so test would fail. testing opt with half suffices.
      if tc.dtype_in == dtypes.bfloat16: continue
      a, b = Tensor.rand(N, N, dtype=tc.dtype_in), Tensor.rand(N, N, dtype=tc.dtype_in)
      r = a.matmul(b, acc_dtype=tc.dtype_out)
      (atol, rtol) = ((0.25, 0.01) if tc.dtype_out == dtypes.half else (3e-2, 1e-3)) if tc.dtype_in == dtypes.half else (1e-4, 1e-4)
      helper_linearizer_opt(r, [
        [],
        [Opt(OptOps.UPCAST, 0, 4)],
        [Opt(OptOps.UPCAST, 1, 4)],
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)], # check upcasts
        [Opt(OptOps.UNROLL, 0, 2)], # check unroll
        [Opt(OptOps.UNROLL, 0, 0)], # check full unroll of reduce with locals
        [Opt(OptOps.LOCAL, 0, 4)], # check local
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 2)], # check combo of unroll and local
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 2)],
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4)],
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.LOCAL, 0, 2)],
        [Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4)], # check permutations
        [Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 0, 4)],
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 4)],
        [Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 4)],
        [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 0, 4)],
        # [Opt(OptOps.GROUP, 0, 2)] # doesn't work because group_for_reduce dims become early locals (conflicting with TC)
      ], apply_tc=True, atol=atol, rtol=rtol)

  @unittest.skip("parallel tensor cores")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_fused_tensor_core_simple(self):
    N = 64
    Tensor.manual_seed(1552)
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if tc.dtype_in == dtypes.bfloat16: continue
      [a, b, c, d] = [Tensor.randn(N, N, dtype=tc.dtype_in).realize() for _ in range(4)]
      r0 = a.matmul(b, acc_dtype=tc.dtype_out)
      r1 = c.matmul(d, acc_dtype=tc.dtype_out)
      check_fused_tc_opt(tc, r0, r1, [a, b, c, d])

  @unittest.skip("parallel tensor cores")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_fused_tensor_core_permuted(self):
    N = 64
    Tensor.manual_seed(1552)
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if tc.dtype_in == dtypes.bfloat16: continue
      # one permuted
      [a, b, c, d] = [Tensor.randn(N, N, dtype=tc.dtype_in).realize() for _ in range(4)]
      r0 = a.matmul(b, acc_dtype=tc.dtype_out)
      r1 = c.T.matmul(d, acc_dtype=tc.dtype_out)
      check_fused_tc_opt(tc, r0, r1, [a, b, c, d])
      # both permuted
      r0 = a.T.matmul(b, acc_dtype=tc.dtype_out)
      r1 = c.T.matmul(d, acc_dtype=tc.dtype_out)
      check_fused_tc_opt(tc, r0, r1, [a, b, c, d])

  def test_padto_matmul(self):
    if CI and Device.DEFAULT in ["AMD", "NV", "CUDA"]: self.skipTest("super slow on CUDA and AMD because of the big grid dims")
    N = 17 * 17
    Tensor.manual_seed(289)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    helper_linearizer_opt(a@b, [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 2, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.PADTO, 2, 32)],
      # can optimize further post PADTO
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 1, 2),],
    ])

  def test_padto_upcasted_not_ok(self):
    N = 4
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    helper_linearizer_opt(a@b, [
      [Opt(OptOps.UPCAST, 0, 0)],
      [Opt(OptOps.UPCAST, 1, 0)],
      [Opt(OptOps.UNROLL, 0, 0)],
      [Opt(OptOps.PADTO, 0, 8)],
      [Opt(OptOps.PADTO, 1, 8)],
      [Opt(OptOps.PADTO, 2, 8)],
    ])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UPCAST, 0, 0), Opt(OptOps.PADTO, 2, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UPCAST, 1, 0), Opt(OptOps.PADTO, 2, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UNROLL, 0, 0), Opt(OptOps.PADTO, 2, 8)]])

  def test_padto_sum_ok(self):
    N = 18 * 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one dimension
    a = Tensor.rand(N, N).shrink(((0, 17), (0, 17))) * 100
    b = (Tensor.rand(N, N) < 0.5).realize().shrink(((0, 17), (0, 17)))

    helper_linearizer_opt(a.sum(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])
    helper_linearizer_opt(a.sum(1), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

    # can pad sum reduce axis if there's no unsafe ops prior to sum
    for axis in (0, 1):
      helper_linearizer_opt(a.sum(), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(a.sum(0), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(0), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(acc_dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(0, acc_dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(1, acc_dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])

    # having unsafe ops after sum is fine
    helper_linearizer_opt(a.sum().exp(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(a.sum(0).exp(), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_sum_not_ok(self):
    N = 18 * 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one dimension
    a = Tensor.rand(N, N).shrink(((0, 17), (0, 17))).exp()
    # exp is not safe to pad
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.exp().sum(), [[Opt(OptOps.PADTO, 0, 32)],])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.exp().sum(0), [[Opt(OptOps.PADTO, 1, 32)],])

    b = a < -1
    # lt is not safe to pad
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(b.sum(), [[Opt(OptOps.PADTO, 0, 32)],])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(b.sum(0), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_max(self):
    N = 18 * 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one axis
    a = -Tensor.rand(N, N).shrink(((0, 17), (0, 17))) * 100

    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])
    helper_linearizer_opt(a.max(1), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

    # cannot pad max kernel on reduce
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.max(), [[Opt(OptOps.PADTO, 0, 32)],])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.max(0), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_where(self):
    Tensor.manual_seed(0)
    N = 17 * 17
    a = (Tensor.randn(N, N).realize().max(axis=0, keepdim=True) > 1).where(1, 0)
    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

  def test_padto_where_multioutput(self):
    Tensor.manual_seed(0)
    N = 17 * 17
    r = Tensor.randn(N, N).realize().max(axis=0, keepdim=True) > 1
    a0 = r.where(1, 0)
    a1 = r.where(2, 0)
    helper_linearizer_opt([a0.max(0), a1.max(0)], [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_padto_group(self):
    Tensor.manual_seed(0)
    ld0 = LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 4, 2, 6, 1, 3), strides=(0, 0, 0, 0, 0, 18, 0, 3, 0, 1), offset=0, mask=None, contiguous=False),)))) # noqa: E501
    ld1 = LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 4, 2, 6, 1, 3), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)))) # noqa: E501
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(ld0, ld1)),), arg=(0, 2, 4, 6)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1, 1, 1, 4, 1, 6, 1, 3), strides=(0, 0, 0, 0, 0, 18, 0, 3, 0, 1), offset=0, mask=None, contiguous=True),)))) # noqa: E501
    data1 = Tensor.randn(2, 1, 4, 1, 3, 4, 2, 6, 1, 3).realize()
    data2 = Tensor.randn(2, 1, 4, 1, 3, 4, 2, 6, 1, 3).realize()
    helper_linearizer_ast((ast, ), [data1, data2], opts=[
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.GROUP, 0, 4)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8), Opt(OptOps.GROUP, 0, 4)]
    ])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skip("AST has implicit movement ops")
  def test_padto_sum_multireduce(self):
    Tensor.manual_seed(0)
    N = 17
    x = Tensor.rand(N, N).realize()
    opts = [[Opt(OptOps.PADTO, 0, 32)],[Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],]
    x_ld = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, ShapeTracker.from_shape((N, N))))

    def ast(axis, output_shape):
      r0 = LazyOp(ReduceOps.SUM, (x_ld,), axis)
      r1 = LazyOp(ReduceOps.SUM, (LazyOp(BinaryOps.ADD, (x_ld, LazyOp(op=UnaryOps.NEG, src=(r0,), arg=None)),),), axis)
      return LazyOp(BufferOps.STORE, (r1, ), MemBuffer(0, dtypes.float, ShapeTracker.from_shape(output_shape))),
    helper_linearizer_ast(ast((0, ), (1, 17)), [x], opts=opts, wanna_output=[(x.numpy()-x.numpy().sum(axis=0,keepdims=True)).sum(0)])
    helper_linearizer_ast(ast((1, ), (17, 1)), [x], opts=opts, wanna_output=[(x.numpy()-x.numpy().sum(axis=1,keepdims=True)).sum(1)])

    expected = (x.numpy()-x.numpy().sum(axis=0,keepdims=True)).sum(0)
    helper_linearizer_ast(ast((0, ), (1, 17)), [x], opts=[[Opt(OptOps.PADTO, 1, 32)]], wanna_output=[expected])

    op = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.ADD, src=(x_ld,LazyOp(op=UnaryOps.NEG, src=(LazyOp(op=ReduceOps.SUM, src=(x_ld,), arg=(0,1)),),arg=None))),), arg=(0,1)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1), strides=(0, 1), offset=0, mask=None, contiguous=True),)))) # noqa: E501
    helper_linearizer_ast((op,), [x], opts=[[Opt(OptOps.PADTO, 0, 32)],], wanna_output=[(x.numpy()-x.numpy().sum(keepdims=True)).sum()])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skip("AST has implicit movement ops")
  def test_padto_max_multireduce(self):
    Tensor.manual_seed(0)
    N = 17
    x = Tensor.rand(N, N).realize()
    opts = [[Opt(OptOps.PADTO, 0, 32)],[Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],]
    x_ld = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, ShapeTracker.from_shape((N, N))))

    def ast(axis, output_shape):
      r0 = LazyOp(ReduceOps.MAX, (x_ld,), axis)
      r1 = LazyOp(ReduceOps.MAX, (LazyOp(BinaryOps.ADD, (x_ld,r0,),),), axis)
      return LazyOp(BufferOps.STORE, (r1, ), MemBuffer(0, dtypes.float, ShapeTracker.from_shape(output_shape))),
    helper_linearizer_ast(ast((0, ), (1, 17)), [x], opts=opts, wanna_output=[(x.numpy()+x.numpy().max(axis=0,keepdims=True)).max(0)])
    helper_linearizer_ast(ast((1, ), (17, 1)), [x], opts=opts, wanna_output=[(x.numpy()+x.numpy().max(axis=1,keepdims=True)).max(1)])

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI doesn't support multiple sync threads yet")
  @unittest.skip("AST has implicit movement ops")
  def test_padto_where_multireduce(self):
    # we need to make sure the ternary operators nest properly
    N = 17
    x = Tensor.rand(N, N).realize()
    a = Tensor.rand(1, 1).realize()
    b = Tensor.rand(1, 1).realize()
    opts = [[Opt(OptOps.PADTO, 0, 32)],[Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],]

    # TODO: these large ASTs are suboptimal but we need this until the scheduler can fuse these
    wanna_output = np.where(0.5*17 < (x.numpy()+np.where(0.75*17 < x.numpy().sum(axis=1,keepdims=True), a.numpy(), b.numpy())).sum(axis=1),0.0,1.0)
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=TernaryOps.WHERE, src=(LazyOp(op=BinaryOps.CMPLT, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.5*17, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape((N,N)))),LazyOp(op=TernaryOps.WHERE, src=(LazyOp(op=BinaryOps.CMPLT, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.75*17, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape((N,N)))),), arg=(1,)))),LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),)),)),), arg=(1,)),)),LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.0, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape((N,1)))) # noqa: E501
    helper_linearizer_ast((ast,), [x,a,b], opts=opts, wanna_output=[wanna_output])

    wanna_output = np.where(0.5*17 < (x.numpy()+np.where(0.75*17 < x.numpy().sum(axis=0,keepdims=True), a.numpy(), b.numpy())).sum(axis=0),0.0,1.0)
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=TernaryOps.WHERE, src=(LazyOp(op=BinaryOps.CMPLT, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.5*17, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape((N,N)))),LazyOp(op=TernaryOps.WHERE, src=(LazyOp(op=BinaryOps.CMPLT, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.75*17, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape((N,N)))),), arg=(0,)))),LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),)),)),), arg=(0,)),)),LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.0, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape((1,N)))) # noqa: E501
    helper_linearizer_ast((ast,), [x,a,b], opts=opts, wanna_output=[wanna_output])

    # pad reduce axis
    helper_linearizer_ast((ast,), [x,a,b], opts=[[Opt(OptOps.PADTO, 1, 32)],], wanna_output=[wanna_output])

    wanna_output = np.where(0.5*17 < (x.numpy()+np.where(0.75*17 < x.numpy().sum(keepdims=True), a.numpy(), b.numpy())).sum(keepdims=True),0.0,1.0)
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=TernaryOps.WHERE, src=(LazyOp(op=BinaryOps.CMPLT, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.5*17, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape((N,N)))),LazyOp(op=TernaryOps.WHERE, src=(LazyOp(op=BinaryOps.CMPLT, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.75*17, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape((N,N)))),), arg=(0,1,)))),LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),)),)),), arg=(0,1,)),)),LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.0, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))),)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape((1,1)))) # noqa: E501
    helper_linearizer_ast((ast,), [x,a,b], opts=[[Opt(OptOps.PADTO, 0, 32)],], wanna_output=[wanna_output.flatten()])

  def test_padto_matmul_multireduce(self):
    if CI and Device.DEFAULT in ["AMD", "NV", "CUDA"]: self.skipTest("super slow on CUDA and AMD because of the big grid dims")
    N = 17 * 17
    Tensor.manual_seed(289)
    a = Tensor.rand(N, N).realize()
    b = Tensor.rand(N, N).realize()
    c = Tensor.rand(N, N).realize()
    d = Tensor.rand(N, N).realize()
    r0 = a@b
    r1 = c@d
    ast = _temp_create_multireduce_ast(r0,r1)
    helper_linearizer_ast(ast, [a,b,c,d], opts=[
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 2, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.PADTO, 2, 32)],
      # can optimize further post PADTO
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 1, 2),],
    ], wanna_output=[(a.numpy()@b.numpy()+c.numpy()@d.numpy()).reshape(N, N, 1)])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_color_shapes_with_local(self):
    N = 32
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    opts_shapes = [
      ([Opt(OptOps.LOCAL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("red",32)]),
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 2)], [("blue",16),("blue",32),("cyan",2),("green",2),("red",16)]),
      # check to ensure local_dims are stable for full UNROLL of first_reduce
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.UNROLL, 0, 0),Opt(OptOps.LOCAL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      # check behavior for full UNROLL on an existing GROUP
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 0),Opt(OptOps.UNROLL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("green",16),("magenta",2)]),
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 0),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.GROUP, 0, 0),Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.GROUP, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",32),("blue",32),("red",16),("magenta",2)]),
    ]
    helper_linearizer_opt(r, [x[0] for x in opts_shapes], color_sizes=[x[1] for x in opts_shapes])

if __name__ == '__main__':
  unittest.main()
