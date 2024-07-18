from __future__ import annotations
from typing import TypeVar, Generic, Callable, List, Tuple, Union, Dict, cast, Optional, Any
import functools, itertools, collections
from tinygrad.tensor import Tensor
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, GRAPH, BEAM, getenv, all_int, GraphException, colored, JIT
from tinygrad.device import Buffer, Compiled, Device
from tinygrad.dtype import DType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable, sint
from tinygrad.engine.realize import ExecItem, capturing, EmptyOp, ViewOp, BufferXfer, CompiledRunner, Runner
from tinygrad.engine.schedule import _internal_memory_planner
from tinygrad.nn.state import get_parameters
from weakref import WeakKeyDictionary

def apply_graph_to_jit(jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]) -> List[ExecItem]:
  # Split JIT cache into batches for faster graph execution.
  # This allows the accelerator to run some batches while subsequent graphs are still being updated.
  max_batch_size = getenv("JIT_BATCH_SIZE", 32)
  graphed_jit_cache: List[ExecItem] = []
  current_batch: List[ExecItem] = []
  current_device: Optional[Compiled] = None

  def flush_batch():
    nonlocal current_batch, current_device, max_batch_size
    try:
      if len(current_batch) <= 1 or current_device is None: raise GraphException("only one kernel doesn't graph")
      graph_runner = current_device.graph(current_batch, input_rawbuffers, var_vals)
      # clear jit inputs to allow their memory to be freed/reused
      for (j,i) in graph_runner.input_replace.keys(): graph_runner.jit_cache[j].bufs[i] = None
      graphed_jit_cache.append(ExecItem(graph_runner, cast(List[Optional[Buffer]], input_rawbuffers)))
      max_batch_size *= 2
      if DEBUG >= 2: print(f"\tJIT GRAPHing batch with {len(current_batch)} kernels on device {current_device}")
    except GraphException as e:
      graphed_jit_cache.extend(current_batch)
      if DEBUG >= 2: print(f"\tJIT GRAPHing failed batch with {len(current_batch)} kernels on device {current_device}: {e}")
    current_batch = []
    current_device = None

  for ji in jit_cache:
    if ji.prg.__class__ in {EmptyOp, ViewOp}: continue
    ji_graph_dev: Optional[Compiled] = None # device on which the ji will be graphed. Not graphed if None.
    if isinstance(ji.prg, CompiledRunner): ji_graph_dev = ji.prg.device
    elif isinstance(ji.prg, BufferXfer) and ji.bufs[0] and ji.bufs[0].device.split(":", 1)[0] in {"CUDA", "NV", "AMD"}:
      ji_graph_dev = Device[ji.bufs[0].device]

    graph_class = (ji_graph_dev.graph.func if isinstance(ji_graph_dev.graph, functools.partial) else ji_graph_dev.graph) if ji_graph_dev else None #type: ignore
    can_be_graphed = ji_graph_dev and ji_graph_dev.graph
    can_share_graph = (ji_graph_dev == current_device or (isinstance(graph_class, type) and issubclass(graph_class, MultiGraphRunner)) and
                       type(ji_graph_dev) is type(current_device))
    can_extend_graph_batch = can_be_graphed and len(current_batch) < max_batch_size and can_share_graph
    if not can_extend_graph_batch and len(current_batch) > 0: flush_batch()

    if can_be_graphed: current_batch.append(ji)
    else: graphed_jit_cache.append(ji)

    current_device = ji_graph_dev

  if len(current_batch) > 0: flush_batch()
  return graphed_jit_cache

def get_input_replace(jit_cache: List[ExecItem], input_rawbuffers:List[Buffer]) -> Dict[Tuple[int, int], int]:
  input_replace: Dict[Tuple[int, int], int] = {}
  for j,ji in enumerate(jit_cache):
    for i,a in enumerate(ji.bufs):
      if a in input_rawbuffers:
        input_replace[(j,i)] = input_rawbuffers.index(a)
  return input_replace

class GraphRunner(Runner):  # pylint: disable=abstract-method
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.jc_idx_with_updatable_launch_dims = []
    self.jc_idx_with_updatable_var_vals = []
    op_estimate: sint = 0
    mem_estimate: sint = 0
    for j,ji in enumerate(jit_cache):
      op_estimate += ji.prg.op_estimate
      mem_estimate += ji.prg.mem_estimate
      if isinstance(ji.prg, CompiledRunner):
        if ji.prg.p.vars: self.jc_idx_with_updatable_var_vals.append(j)
        if (ji.prg.p.global_size and not all_int(ji.prg.p.global_size)) or (ji.prg.p.local_size and not all_int(ji.prg.p.local_size)):
          self.jc_idx_with_updatable_launch_dims.append(j)
    self.vars = sorted(var_vals.keys(), key=lambda v: v.expr)
    super().__init__(colored(f"<batched {len(self.jit_cache)}>", "cyan"), jit_cache[0].prg.dname.split(":")[0], op_estimate, mem_estimate)

class MultiGraphRunner(GraphRunner):  # pylint: disable=abstract-method
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    self.w_dependency_map: Dict[Any, Any] = {}
    self.r_dependency_map: Dict[Any, List[Any]] = collections.defaultdict(list)
    super().__init__(jit_cache, input_rawbuffers, var_vals)

  def _access_resources(self, read, write, new_dependency:Any):
    # To synchronize access to resources, we monitor the necessary prerequisites for accessing each resource,
    # whether for write or read operations. A resource can be accessed by either a single writer or multiple readers.
    wait_nodes = []

    for rawbuf in read + write:
      if id(rawbuf.base._buf) in self.w_dependency_map: wait_nodes.append(self.w_dependency_map[id(rawbuf.base._buf)])
    for rawbuf in write:
      if id(rawbuf.base._buf) in self.r_dependency_map: wait_nodes.extend(self.r_dependency_map.pop(id(rawbuf.base._buf)))

    for rawbuf in read: self.r_dependency_map[id(rawbuf.base._buf)].append(new_dependency)
    for rawbuf in write: self.w_dependency_map[id(rawbuf.base._buf)] = new_dependency
    return list({id(x):x for x in wait_nodes}.values())

ReturnType = TypeVar('ReturnType')
class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn
    self.reset()

  def add_buffer(self, b:Buffer) -> Buffer:
    if found:=self.buffer_replace.get(b, None): return found
    if b.is_allocated() or b.lb_refcount > 0: return b
    if b._base is not None:
      self.buffer_replace[b] = ret = Buffer(b.device, b.size, b.dtype, base=self.add_buffer(b._base), offset=b.offset)
    else:
      self.buffer_replace[b] = ret = Buffer(b.device, b.size, b.dtype, options=b.options)
    return ret

  def add(self, ei:ExecItem):
    self.jit_cache.append(ExecItem(ei.prg, [self.add_buffer(buf) for buf in ei.bufs if buf is not None]))

  def reset(self):
    self.jit_cache: List[ExecItem] = []
    self.input_replace: Dict[Tuple[int, int], int] = {}
    self.extra_view_inputs: List[Tuple[int, int, str, int, DType]] = []
    self.buffer_replace: WeakKeyDictionary[Buffer, Buffer] = WeakKeyDictionary()
    self.cnt: int = 0

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_tensors: List[Tuple[Union[int, str], Tensor]] = \
      [(cast(Union[int, str], name),t) for name,t in itertools.chain(enumerate(args), sorted(kwargs.items())) if t.__class__ is Tensor]
    if input_tensors: Tensor.realize(*[t for _,t in input_tensors])
    names: List[Union[int, str]] = [name for name,_ in input_tensors]
    lbs: List[LazyBuffer] = flatten([t.lazydata.lbs for _,t in input_tensors])
    st_varvals_dtype_device = [(*lb.st.unbind(), lb.dtype, lb.device) for lb in lbs]
    input_buffers: List[Buffer] = [lb.base.realized for lb in lbs if lb.base.realized is not None]
    assert len(set(input_buffers)) == len(input_buffers), "duplicate inputs to JIT"
    var_vals: Dict[Variable, int] = merge_dicts([varvals for _,varvals,_,_ in st_varvals_dtype_device] + \
                                                [dict(v.unbind() for v in itertools.chain(args, kwargs.values()) if isinstance(v, Variable))])
    st_vars_dtype_device = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in st_varvals_dtype_device]
    if not JIT or self.cnt == 0:
      # jit ignore
      with Context(BEAM=0 if getenv("IGNORE_JIT_FIRST_BEAM") else BEAM.value):
        self.ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(self.ret)): Tensor.realize(params[0], *params[1:])
    elif self.cnt == 1:
      # jit capture
      self.expected_names: List[Union[int, str]] = names
      self.expected_st_vars_dtype_device: List[Tuple[ShapeTracker, Tuple[Variable, ...], DType, str]] = st_vars_dtype_device
      if capturing: raise RuntimeError(f"having TinyJit inside another TinyJit is not supported {len(capturing)=} {capturing=}")
      with Context(GRAPH=getenv("JITGRAPH", GRAPH.value), BEAM=getenv("JITBEAM", BEAM.value)):
        capturing.append(self)
        try:
          self.ret = self.fxn(*args, **kwargs)
          if len(params:=get_parameters(self.ret)): Tensor.realize(params[0], *params[1:])
        except Exception as e: raise e
        finally: capturing.clear()
      del self.buffer_replace
      assert len(self.jit_cache), "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_buffers)} inputs")

      # track inputs that are views of buffers
      for item in self.jit_cache:
        for b in item.bufs:
          if b is not None and b._base is not None and b._base in input_buffers:
            input_buffers.append(b)
            self.extra_view_inputs.append((input_buffers.index(b.base), b.offset, b.device, b.size, b.dtype))

      # memory planning (optional)
      # Exclude buffers involved in transfer ops to preserve parallelism.
      noopt_buffers = {b for ji in self.jit_cache if isinstance(ji.prg, BufferXfer) for b in ji.bufs}
      assigned = _internal_memory_planner([cast(List[Buffer], item.bufs) for item in self.jit_cache], noopt_buffers, debug_prefix="JIT ")
      self.jit_cache = [ExecItem(item.prg, [assigned.get(b,b).ensure_allocated() for b in item.bufs if b is not None]) for item in self.jit_cache]

      # Condense the items into a graph executor.
      if JIT < 2: self.jit_cache = apply_graph_to_jit(self.jit_cache, input_buffers, var_vals)

      self.input_replace = get_input_replace(self.jit_cache, input_buffers)
      if DEBUG >= 1 and len(set(self.input_replace.values())) != len(input_buffers): print("WARNING: some input tensors not found")
    elif self.cnt >= 2:
      # jit exec
      assert self.expected_names == names, f"args mismatch in JIT: {self.expected_names=} != {names}"
      assert self.expected_st_vars_dtype_device == st_vars_dtype_device, \
        f"args mismatch in JIT: {self.expected_st_vars_dtype_device=} != {st_vars_dtype_device=}"
      for idx, offset, device, size, dtype in self.extra_view_inputs:
        input_buffers.append(Buffer(device, size, dtype, base=input_buffers[idx], offset=offset).ensure_allocated())
      for (j,i),input_idx in self.input_replace.items(): self.jit_cache[j].bufs[i] = input_buffers[input_idx]
      if DEBUG >= 1 and len(self.jit_cache) >= 10: print(f"jit execs {len(self.jit_cache)} kernels")
      for ei in self.jit_cache: ei.run(var_vals, jit=True)

    # clear jit inputs
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].bufs[i] = None

    self.cnt += 1
    return self.ret
