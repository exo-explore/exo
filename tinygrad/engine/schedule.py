import sys, pickle, atexit
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set, DefaultDict, Union, cast, get_args
from tinygrad.ops import MetaOps, BufferOps, LazyOp, Op, ReduceOps, ConstBuffer, MemBuffer, UNSAFE_PAD_OPS, UnaryOps, reduce_st
from tinygrad.engine.graph import log_lazybuffer, realized_lazybuffer
from tinygrad.helpers import GRAPH, DEBUG, MULTIOUTPUT, SAVE_SCHEDULE, GlobalCounters, colored, prod, dedup, all_int, merge_dicts, getenv, Metadata
from tinygrad.shape.symbolic import Variable
from tinygrad.dtype import ConstType, ImageDType, dtypes
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer, Device

# creation can recurse a lot
sys.setrecursionlimit(10000)

# optionally log the ops to disk
logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None

# *** ScheduleItem return type ***

@dataclass(frozen=True)
class ScheduleItem:
  ast: LazyOp
  bufs: Tuple[Buffer, ...]
  metadata: Optional[List[Metadata]] = None
  @property
  def outputs(self) -> Tuple[Buffer, ...]:
    """Read/write or write only buffers in the schedule."""
    return self.bufs[:len(self.ast.src)] if self.ast.op is MetaOps.KERNEL else self.bufs[0:1]
  @property
  def inputs(self) -> Tuple[Buffer, ...]:
    """Read only buffers in the schedule."""
    return self.bufs[len(self.ast.src):] if self.ast.op is MetaOps.KERNEL else self.bufs[1:]

# *** DAG transformation: List[LazyBuffer] -> ScheduleItem ***

def _recursive_lazyop(buf:LazyBuffer, inputs:List[LazyBuffer], outputs:Tuple[LazyBuffer, ...], var_vals:Dict[Variable, int], st:ShapeTracker,
                      realizes:Dict[LazyBuffer, None], assign_targets:Dict[LazyBuffer, LazyBuffer],
                      reduce_info:Dict[LazyBuffer, Tuple[ShapeTracker, Tuple[int, ...]]], cache) -> LazyOp:
  """recursively create a lazyop"""
  if buf is not buf.base: st, buf = buf.st+st, buf.base
  if (buf, st) in cache: return cache[(buf, st)]
  arg = buf.arg

  # consts are always fused and generated
  if buf.op is MetaOps.CONST:
    unbound_st, st_var_vals = st.simplify().unbind()
    var_vals.update(st_var_vals)
    if isinstance(buf.arg, Variable):
      val, var_val = buf.arg.unbind()
      var_vals.__setitem__(val, var_val)
    else:
      assert isinstance(buf.arg, get_args(ConstType)), f"cannot create ConstBuffer with value {buf.arg}"
      val = buf.arg
    return LazyOp(BufferOps.CONST, (), ConstBuffer(val, buf.dtype, unbound_st))

  # if we aren't fusing it, it's a load and we add it to the inputs
  if buf.realized is not None or (buf in realizes and buf not in outputs):
    unbound_st, st_var_vals = st.simplify().unbind()
    var_vals.update(st_var_vals)
    if buf in assign_targets:
      # can only assign to contiguous read+write buffer
      if not unbound_st.contiguous:
        # we also allow masked views. if it has a single view and it's equal when you shrink a contig, it's fine
        if not (len(unbound_st.views) == 1 and unbound_st.views[0].mask is not None and
            ShapeTracker.from_shape(unbound_st.shape).shrink(unbound_st.views[0].mask) == unbound_st.shrink(unbound_st.views[0].mask)):
          raise RuntimeError("self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n"
                             +colored("   - a += a.T\n", "red")+colored("   + a += a.T.contiguous()", "green"))
      return LazyOp(BufferOps.LOAD, (), MemBuffer(outputs.index(assign_targets[buf]), buf.dtype, unbound_st))
    if buf not in inputs: inputs.append(buf)
    return LazyOp(BufferOps.LOAD, (), MemBuffer(len(outputs)+inputs.index(buf), buf.dtype, unbound_st))

  # if a CONTIGUOUS or ASSIGN made it all the way here, just skip it
  if buf.op is MetaOps.CONTIGUOUS:
    assert buf in outputs
    return _recursive_lazyop(buf.srcs[0], inputs, outputs, var_vals, st, realizes, assign_targets, reduce_info, cache)
  if buf.op is MetaOps.ASSIGN:
    assert buf in outputs
    assert buf.srcs[1].base is buf.srcs[1], "assign must be to base"
    assert buf.srcs[1].realized is not None, f"assign must be already realized to schedule {buf.srcs[1]}"
    return _recursive_lazyop(buf.srcs[0], inputs, outputs, var_vals, st, realizes, assign_targets, reduce_info, cache)

  # if it's a reduce, we have to change the shapetracker
  if buf.op in ReduceOps:
    assert st.contiguous, "ReduceOps late fusion must be contiguous"
    st, arg = reduce_info[buf]

  # otherwise we fuse it like normal
  return cache.setdefault((buf, st), LazyOp(cast(Op,buf.op), tuple(_recursive_lazyop(x, inputs, outputs, var_vals, st, realizes, assign_targets, \
      reduce_info, cache) for x in buf.srcs), arg))

def _recurse_reduceops(buf:LazyBuffer, st:ShapeTracker, realizes:Dict[LazyBuffer, None], outs:List[LazyBuffer], reduce_info:Dict, cache):
  if buf.base.realized is not None or (buf.base in realizes and buf.base not in outs) or (buf, st) in cache: return
  cache.setdefault((buf, st))
  if buf is not buf.base: st, buf = buf.st+st, buf.base
  for x in buf.srcs: _recurse_reduceops(x, buf.srcs[0].st if buf.op in ReduceOps else st, realizes, outs, reduce_info, cache)
  if buf.op in ReduceOps:
    reduce_input, axis = buf.srcs[0], buf.arg
    assert st.contiguous
    st = ShapeTracker.from_shape(reduce_input.shape)
    reduce_info[buf] = (st, axis)

def _lower_lazybuffer(outs:List[LazyBuffer], realizes:Dict[LazyBuffer, None]):
  """describe the computation for a LazyBuffer with LazyOp + inputs + var_vals"""
  if (out:=outs[0]).op is MetaOps.COPY and getenv("USE_COPY_KERNEL") and out.device.split(":")[0] == out.srcs[0].device.split(":")[0]:
    rd = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.uint8, st:=ShapeTracker.from_shape((out.arg,))))
    return LazyOp(MetaOps.KERNEL, (LazyOp(BufferOps.STORE, (rd,), MemBuffer(0, dtypes.uint8, st)), )), [x.base for x in out.srcs], {}, []
  if out.op in {MetaOps.CUSTOM, MetaOps.COPY, MetaOps.EMPTY, MetaOps.VIEW}: return LazyOp(out.op, (), out.arg), [x.base for x in out.srcs], {}, []
  var_vals: Dict[Variable, int] = merge_dicts([out.st.var_vals.copy() for out in outs])
  assign_targets = {x.srcs[1]:x for x in outs if x.op is MetaOps.ASSIGN}
  cache: Dict[Tuple[LazyBuffer, ShapeTracker], LazyOp] = {}
  ast: List[LazyOp] = []
  inputs: List[LazyBuffer] = []
  reduce_info: Dict[LazyBuffer, Tuple[ShapeTracker, Tuple[int, ...]]] = {}
  seen_ops: Dict[Tuple[LazyBuffer, ShapeTracker], None] = {}
  for i, out in enumerate(outs):
    _recurse_reduceops(out, out.st, realizes, outs, reduce_info, seen_ops)
    output_st = ShapeTracker.from_shape(reduce_st(*deque(reduce_info.values(), 1).pop()) if reduce_info else out.shape)
    output_view = out.arg[0] if out.op is MetaOps.ASSIGN and out.arg else output_st
    lop = _recursive_lazyop(out, inputs, tuple(outs), var_vals, output_st, realizes, assign_targets, reduce_info, cache=cache)
    output_view, vv = output_view.simplify().unbind()
    if vv: var_vals.update(vv)
    ast.append(LazyOp(BufferOps.STORE, (lop, ), MemBuffer(i, out.dtype, output_view)))
  return LazyOp(MetaOps.KERNEL, tuple(ast)), inputs, var_vals, dedup([x[0].metadata for x in cache if x[0].metadata and x[0] not in inputs])

# *** DAG creation: decide which LazyBuffers should realize ***

def _recurse_lb(buf:LazyBuffer, realizes:Dict[LazyBuffer, None], allbufs:Dict[LazyBuffer, None],
                simple_pads:Set[LazyBuffer], children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], scheduled=False):
  """recursively search the entire graph for all LazyBuffers, insert realizes after expands"""
  if buf in allbufs or buf.base.realized is not None: return
  if GRAPH: log_lazybuffer(buf, scheduled)
  # view
  if buf.base != buf:
    # fuse some pads
    if len(buf.st.views) == 1 and buf.st.views[-1].mask is not None and all_int(buf.base.st.shape) and \
        prod(buf.base.st.shape) >= prod([y-x for x,y in buf.st.views[-1].mask]):
      simple_pads.add(buf.base)
    # realize all expands
    elif prod(buf.base.st.shape) < prod(buf.st.shape):
      if buf.base.op in ReduceOps and buf.base.srcs[0].base.op is MetaOps.CONST:
        pass # don't realize reduceops on const (unless base is forced_realize)
      # this was causing "test_lil_model" to fail
      if buf.base.op is UnaryOps.CAST and isinstance(buf.base.srcs[0].dtype, ImageDType) and isinstance(buf.base.arg, ImageDType):
        simple_pads.add(buf.base) # don't realize image to image casts. this is part of a larger problem
      else:
        realizes[buf.base] = None
    # check all other pads for safe fusion
    elif any(v.mask is not None for v in buf.st.views): simple_pads.add(buf.base)
    return _recurse_lb(buf.base, realizes, allbufs, simple_pads, children)
  # base
  allbufs[buf] = None
  if buf.forced_realize: realizes[buf] = None
  if buf.op in MetaOps: realizes[buf.base] = None
  if buf.op is MetaOps.COPY:
    assert buf.srcs[0].st.contiguous and buf.srcs[0].size == buf.srcs[0].base.size, "can only copy contig"
    realizes[buf.srcs[0].base] = None
  if buf.op is MetaOps.VIEW: realizes[buf.srcs[0].base] = None
  for x in buf.srcs:
    if x.base.realized is None: children[x.base][buf] = None
    _recurse_lb(x, realizes, allbufs, simple_pads, children)

def _is_padding_okay(buf:LazyBuffer, realizes:Dict[LazyBuffer, None]) -> bool:
  if buf in realizes or buf.realized is not None: return True
  # NOTE: this broke to_image_idx and coder with JIT
  if buf.op in UNSAFE_PAD_OPS: return False
  return all(_is_padding_okay(x.base, realizes) for x in buf.srcs)

def _recursive_group(tr:LazyBuffer, st:ShapeTracker, r:LazyBuffer, children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]],
                     realizes:Dict[LazyBuffer, None], reduce_for_op:Dict[LazyBuffer, LazyBuffer], group:Set[LazyBuffer], cache:Set):
  """recursively search the LazyBuffer for groupable children, realize the LazyBuffer if a child can't group"""
  if (tr, st) in cache: return
  cache.add((tr, st))
  if tr in realizes:
    # can only fuse contiguous
    # max one reduceop per kernel
    if not st.contiguous or st.size != r.st.size or tr in reduce_for_op: group.add(r)
    return group.add(tr)
  for tr_next in children[tr]:
    # max one reduceop per kernel
    if tr_next.op in ReduceOps: return group.add(r)
    # can only fuse contiguous
    if len(st_childs:=dedup(s for s in tr_next.srcs if s.base == tr)) > 1: return group.add(r)
    _recursive_group(tr_next, st+st_childs[0].st, r, children, realizes, reduce_for_op, group, cache)

def _graph_schedule(outs:List[LazyBuffer], seen:Set[LazyBuffer]):
  """create a graph for realizing the outputs"""
  # start by just realizing the buffers passed in
  realizes: Dict[LazyBuffer, None] = {x.base:None for x in outs if x.base.realized is None}
  allbufs: Dict[LazyBuffer, None] = {}
  simple_pads: Set[LazyBuffer] = set()
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  for out in outs: _recurse_lb(out.base, realizes, allbufs, simple_pads, children, scheduled=True)
  assign_targets = {x.srcs[1]:x for x in realizes if x.op is MetaOps.ASSIGN and x not in seen and x.realized is None}

  # check if we have to realize pads
  for p in simple_pads:
    if not _is_padding_okay(p, realizes):
      realizes[p] = None

  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[LazyBuffer, LazyBuffer] = {}
  for r in allbufs:
    if r.op not in ReduceOps or r in realizes: continue

    group: Set[LazyBuffer] = set()
    _recursive_group(r, r.st, r, children, realizes, reduce_for_op, group, cache=set())
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    if not forced_realize and len(group) > 1:
      # create a multi output kernel if the LazyBuffers can cleanly group
      cache: Set[LazyBuffer] = set()
      rc_parents, rc_children = deque(group), deque(group)
      while rc_parents:
        if (p:=rc_parents.pop()) in cache: continue
        cache.add(p)
        # max one reduceop per kernel
        if p.op in ReduceOps:
          forced_realize = True
          break
        rc_parents.extend(x.base for x in p.srcs if x.base.realized is None and x.base is not r)
      # search descendants of the reduceop that can cleanly group
      cache.clear()
      realized_descendants: Set[LazyBuffer] = set()
      while rc_children and not forced_realize:
        if (c:=rc_children.pop()) in cache: continue
        cache.add(c)
        if c.op in ReduceOps or not c.st.contiguous or c.st.size != r.st.size or c in reduce_for_op:
          realized_descendants.clear()
          break
        if c in realizes and c not in group: realized_descendants.add(c)
        rc_children.extend(x for x in children[c] if x.realized is None and x.device == r.device)
      group.update(realized_descendants)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x.op is MetaOps.ASSIGN for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p:=parents.pop().base).realized or p in realizes:
          if p in assign_targets and assign_targets[p] not in group: forced_realize, can_chase = True, False
          continue
        parents.extend(p.srcs)
    if forced_realize and (r.srcs[0].base.op is not MetaOps.CONST or any(x.shape != r.shape for x in children[r])):
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = tr.st
        while len(children[tr]) == 1:
          tr_next = next(iter(children[tr]))
          st_childs = dedup(s for s in tr_next.srcs if s.base is tr)
          if len(st_childs) > 1: break
          if st.size != st_childs[0].st.size: break
          st = st + st_childs[0].st
          if not st.contiguous or tr_next.op in ReduceOps: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if tr.op is UnaryOps.CAST and tr.arg.itemsize > tr.srcs[0].dtype.itemsize:
          tr = tr.srcs[0].base
        reduce_for_op[tr] = r
      realizes[tr] = None
    else: reduce_for_op.update((tr, r) for tr in group)

  output_groups: DefaultDict[LazyBuffer, List[LazyBuffer]] = defaultdict(list)
  for buf in realizes:
    if buf.realized is not None or buf.op is MetaOps.CONST or buf in seen: continue
    output_groups[reduce_for_op[buf] if buf in reduce_for_op and MULTIOUTPUT else buf].append(buf)

    # make things that can't be images not images
    if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or
                                              not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
      if DEBUG >= 2: print(f"forcing image {buf.dtype} with shape {buf.shape} to float32")
      buf.dtype = dtypes.float32
      # hack the underlying buffer too
      if buf.base is buf:
        assert not hasattr(buf.buffer, '_buf'), "can't fixup allocated buffer"
        buf.buffer.dtype = dtypes.float32
        buf.buffer.options = None

  # preschedule all buffers in realizes
  prescheduled = {group[0]:(group, *_lower_lazybuffer(group, realizes)) for group in output_groups.values()}
  schedule_targets = {out:ps for ps in prescheduled.values() for out in ps[0]}

  graph: DefaultDict[LazyBuffer, List[LazyBuffer]] = defaultdict(list)
  in_degree: DefaultDict[LazyBuffer, int] = defaultdict(int)
  for key, lsi in prescheduled.items():
    if key not in in_degree: in_degree[key] = 0
    # realize outputs after all parents are realized
    scheduled_parents = set(schedule_targets[x][0][0] for x in lsi[2] if x in schedule_targets)
    for x in scheduled_parents:
      graph[x].append(key)
      in_degree[key] += 1
    # realize outputs before a parent is assigned to
    parents_assigns = set(schedule_targets[assign_targets[x]][0][0] for x in lsi[2] if x in assign_targets)
    for assign in parents_assigns:
      graph[key].append(assign)
      in_degree[assign] += 1

  return graph, in_degree, prescheduled

# *** DAG ordering: breadth first search ***

SCHEDULES: List = []
def create_schedule_with_vars(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  if seen is None: seen = set()
  graph, in_degree, prescheduled = _graph_schedule(outs, seen)
  queue = deque(si for key, si in prescheduled.items() if in_degree[key] == 0)
  schedule: List[ScheduleItem] = []
  var_vals: Dict[Variable, int] = {}
  kernel_number = GlobalCounters.kernel_count
  while queue:
    ps = queue.popleft()
    for buf in ps[0]: seen.add(buf)
    if GRAPH:
      kernel_number += 1
      for out in ps[0]: realized_lazybuffer(out, kernel_number)
    var_vals = merge_dicts([var_vals, ps[3]])
    for out in ps[0]: del out.srcs  # can only schedule once
    schedule.append(si:=ScheduleItem(ps[1], tuple(x.buffer for x in ps[0]+ps[2] if x.size != 0), ps[4]))
    if logops and si.ast.op is MetaOps.KERNEL and not any(i.device.startswith("DISK:") for i in si.inputs): logops.write(str(si.ast)+"\n")
    for x in graph[ps[0][0]]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(prescheduled[x])

  if SAVE_SCHEDULE:
    def _save():
      print(f"saving {len(SCHEDULES)} schedule graphs to", fp:=getenv("SAVE_SCHEDULE_PATH", "schedule.pkl"))
      with open(fp, "wb") as f: pickle.dump(SCHEDULES, f)
    if len(SCHEDULES) == 0: atexit.register(_save)
    SCHEDULES.extend((ps[1] for ps in prescheduled.values()) if getenv("CAPTURE_AST") else [(graph, prescheduled)])
  # confirm everything was scheduled correctly
  if not all(degree == 0 for degree in in_degree.values()) or len(prescheduled) != len(schedule):
    raise RuntimeError(f"cycle detected in graph, prescheduled {len(prescheduled)} but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  return schedule, var_vals

def create_schedule(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
  schedule, var_vals = create_schedule_with_vars(outs, seen)
  assert len(var_vals) == 0
  return schedule

# *** memory planning ***

def _internal_memory_planner(buffers:List[Union[List[Buffer], Tuple[Buffer, ...]]], noopt_buffers=None, debug_prefix="") -> Dict[Buffer, Buffer]:
  if getenv("NO_MEMORY_PLANNER"): return {}
  first_appearance, last_appearance = {}, {}
  for i,u in enumerate(buffers):
    for buf in u:
      if buf.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers): continue
      if buf.base not in first_appearance: first_appearance[buf.base] = i
      last_appearance[buf.base] = i

  # Sort buffers by size in descending order, prioritizing largest buffers for allocation first.
  # Track free segments, each containing (start, stop, and buffer that could be reused on this segment).
  free_segs: Dict[Tuple, List[Tuple[int, int, Buffer]]] = defaultdict(list) # Dict[buffer key, Tuple[start, end, buffer to reuse on the seg]]
  def find_replace_buffer(buf, st, en):
    key = (buf.device, buf.dtype, buf.options) + ((buf.nbytes,) if not hasattr(Device[buf.device].allocator, "offset") else tuple())

    default_buf = (0, len(buffers) - 1, buf) # will return the buffer itself if the replace one is not found.
    seg_st, seg_en, seg_buf = next((free_segs[key].pop(i) for i,(sst,sen,_) in enumerate(free_segs[key]) if sst <= st and en <= sen), default_buf)

    free_segs[key] += [(seg_st, st - 1, seg_buf)] if st - 1 >= seg_st else []
    free_segs[key] += [(en + 1, seg_en, seg_buf)] if seg_en >= en + 1 else []

    return seg_buf if seg_buf.nbytes == buf.nbytes else Buffer(buf.device, buf.size, buf.dtype, base=seg_buf)

  buffer_requests = sorted([(first_appearance[buf], last_appearance[buf], buf) for buf in first_appearance.keys()], key=lambda x: -x[2].nbytes)
  assigned = {buf:find_replace_buffer(buf, st, en) for st, en, buf in buffer_requests}

  for i,u in enumerate(buffers):
    for buf in u:
      if buf.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers): continue
      if buf._base is not None: assigned[buf] = Buffer(buf.device, buf.size, buf.dtype, base=assigned.get(buf.base, buf.base).base, offset=buf.offset)
      else: assigned[buf] = assigned.get(buf, buf)

  if DEBUG >= 1 and len(ak:=dedup(x for x in assigned.keys() if x._base is None)) != len(av:=dedup(x for x in assigned.values() if x._base is None)):
    print(debug_prefix+f"memory reduced from {sum([x.nbytes for x in ak])/1e6:.2f} MB -> {sum([x.nbytes for x in av])/1e6:.2f} MB,",
          f"{len(ak)} -> {len(av)} bufs")
  return assigned

def memory_planner(schedule:List[ScheduleItem]) -> List[ScheduleItem]:
  # Exclude buffers involved in load ops (e.g transfers) to preserve parallelism in graphs.
  assigned = _internal_memory_planner([si.bufs for si in schedule],
                                      noopt_buffers={b for si in schedule if si.ast.op is not MetaOps.KERNEL for b in si.bufs})
  return [ScheduleItem(si.ast, tuple(assigned.get(x, x) for x in si.bufs), si.metadata) for si in schedule]
