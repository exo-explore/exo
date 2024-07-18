import unittest

from tinygrad.codegen.kernel import Opt, OptOps
from tinygrad.codegen.kernel import Kernel
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.search import time_linearizer, bufs_from_lin, actions, beam_search
from tinygrad.device import Device, Buffer
from tinygrad.ops import LazyOp, MetaOps, BufferOps, ReduceOps, BinaryOps, MemBuffer, ConstBuffer
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, GlobalCounters
from tinygrad.engine.realize import capturing
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

class TestTimeLinearizer(unittest.TestCase):
  def test_reasonable_time(self):
    si = [i for i in create_schedule([Tensor([1,2,3,4]).add(1).lazydata]) if i.ast.op is MetaOps.KERNEL][0]
    out = Buffer(Device.DEFAULT, si.outputs[0].size, si.outputs[0].dtype).allocate()
    memops = {x.arg.idx:x.arg.st.real_size() for x in si.ast.lazyops if x.op is BufferOps.LOAD}
    rawbufs = [out] + [Buffer(Device.DEFAULT, memops[i], x.dtype).allocate() for i,x in enumerate(si.inputs, start=len(si.outputs))]
    tm = time_linearizer(Kernel(si.ast), rawbufs, allow_test_size=False, cnt=10, disable_cache=True)
    assert tm > 0 and tm != float('inf')

  def test_bufs_from_lin(self):
    si = [i for i in create_schedule([Tensor([1,2,3,4]).add(1).lazydata]) if i.ast.op is MetaOps.KERNEL][0]
    rawbufs = bufs_from_lin(lin:=Kernel(si.ast))
    assert len(rawbufs) == len(lin.membufs)
    assert all(r is not None for r in rawbufs)
    assert all(isinstance(r, Buffer) for r in rawbufs)
    assert all(r.size > 0 for r in rawbufs)

  def test_kernel_count(self):
    """
    Ensure that the kernel count is not incremented by time_linearizer when clearing l2
    """
    # ast of Tensor.zeros(16).contiguous().realize()
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(16,), strides=(0,), offset=0, mask=None, contiguous=False),)))),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(16,), strides=(1,), offset=0, mask=None, contiguous=True),))))  # noqa: E501
    lin = Kernel(ast)
    bufs = bufs_from_lin(lin)

    kernel_count = GlobalCounters.kernel_count
    time_linearizer(lin, bufs, allow_test_size=False, cnt=2, disable_cache=True, clear_l2=True)
    assert GlobalCounters.kernel_count == kernel_count, "kernel count was incremented by time_linearizer"

class TestBEAM(unittest.TestCase):
  def test_dynamic_beam(self):
    # TODO: make this infra globally usable
    class Capture:
      def __init__(self): self.captured = []
      def add(self, x): self.captured.append(x)

    capturing.append(Capture())
    kernel_count = GlobalCounters.kernel_count
    with Context(BEAM=1): Tensor.zeros(16).contiguous().realize()
    assert GlobalCounters.kernel_count == kernel_count + 1
    k_beam_1 = capturing[0].captured
    capturing.clear()

    capturing.append(Capture())
    kernel_count = GlobalCounters.kernel_count
    with Context(BEAM=0): Tensor.zeros(16).contiguous().realize()
    assert GlobalCounters.kernel_count == kernel_count + 1
    k_beam_0 = capturing[0].captured
    capturing.clear()
    self.assertNotEqual(k_beam_0[-1].prg.p.src, k_beam_1[-1].prg.p.src)

  def test_get_kernel_actions(self):
    from test.test_linearizer import helper_realized_ast
    a = Tensor.rand(4, 3)
    b = Tensor.rand(3)
    realized_ast, _ = helper_realized_ast(a @ b)
    from tinygrad.engine.search import get_kernel_actions
    lins = get_kernel_actions(Kernel(realized_ast), False).values()

    # ensure amt=0 are not duplicated
    if Opt(OptOps.UPCAST, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.UPCAST, axis=0, amt=4)]) == 0, "did not de-dup UPCAST"
    if Opt(OptOps.LOCAL, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.LOCAL, axis=0, amt=4)]) == 0, "did not de-dup LOCAL"
    if Opt(OptOps.UNROLL, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.UNROLL, axis=0, amt=3)]) == 0, "did not de-dup UNROLL"
    if Opt(OptOps.GROUP, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.GROUP, axis=0, amt=3)]) == 0, "did not de-dup GROUP"
    if Opt(OptOps.GROUPTOP, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.GROUPTOP, axis=0, amt=3)]) == 0, "did not de-dup GROUPTOP"

  def test_filter_global_buffer(self):
    # taken from https://github.com/tinygrad/tinygrad/issues/4612
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.MAX, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=0, mask=((0, 64128),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))))), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-64128, mask=((64128, 128256),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False)))))), arg=None), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-128256, mask=((128256, 192384),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False)))))), arg=None), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=4, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-192384, mask=((192384, 256512),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False)))))), arg=None), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=5, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-256512, mask=((256512, 320640),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False)))))), arg=None), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=6, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-320640, mask=((320640, 384768),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False)))))), arg=None), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.4285714285714286, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 501, 256), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=(1,)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 256), strides=(0, 0, 1), offset=0, mask=None, contiguous=True),))))  # noqa: E501
    lin = Kernel(ast)

    bufs = bufs_from_lin(lin)
    best_lin = beam_search(lin, bufs, 3)
    assert best_lin
    # need disable_cache to trigger.
    tm = time_linearizer(best_lin, bufs, allow_test_size=False, cnt=2, disable_cache=True)
    assert tm

if __name__ == '__main__':
  unittest.main()
