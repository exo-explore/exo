import itertools
import numpy as np
from typing import DefaultDict, Dict, List, Set, Tuple, TypeVar, Union
from tinygrad.device import Buffer
from tinygrad.engine.realize import CustomOp, capturing, lower_schedule_item
from tinygrad.helpers import DEBUG, MULTIOUTPUT, colored, getenv
from tinygrad.lazy import LazyBuffer
from tinygrad.engine.schedule import _graph_schedule, ScheduleItem
from tinygrad.ops import MetaOps
from tinygrad.tensor import Tensor, _to_np_dtype

ctx_vars = { MULTIOUTPUT: (0, 1) }
FUZZ_SCHEDULE_MAX_PATHS = getenv("FUZZ_SCHEDULE_MAX_PATHS", 10)

def fuzz_schedule(outs:List[LazyBuffer]):
  # find toposorts across all tunable params
  unique_ts: Dict[Tuple[LazyBuffer, ...], Tuple[Dict, Dict[LazyBuffer, Tuple]]] = {}
  for combination in itertools.product(*ctx_vars.values()):
    for var, val in zip(ctx_vars, combination): var.value = val
    graph, in_degree, prescheduled = _graph_schedule(outs, set())
    for ts in find_all_toposorts(graph, in_degree): unique_ts[ts] = (dict(zip([v.key for v in ctx_vars], combination)), prescheduled)
  toposorts = list(unique_ts.items())
  if DEBUG >= 1: print(colored(f"fuzzing {len(toposorts)} schedule permutations", "yellow"))

  # setup ground truth
  ground_truth: Dict[LazyBuffer, memoryview] = {}
  assign_targets: Dict[LazyBuffer, LazyBuffer] = {}
  # IMPORTANT: freeze prerealized bufs before ScheduleItem exec
  prerealized: Dict[LazyBuffer, memoryview] = {}
  seed = Tensor._seed
  ts, (_, prescheduled) = toposorts[0]
  for key in ts:
    for out in (ps:=prescheduled[key])[0]:
      # freeze assign state before exec
      if out.op is MetaOps.ASSIGN:
        prerealized[out] = out.buffer.as_buffer()
        assign_targets[out.srcs[1]] = out
    for x in ps[2]:
      if x not in ground_truth and x.device != "NPY": prerealized[x] = x.buffer.as_buffer()
    si = ScheduleItem(ps[1], tuple(x.buffer for x in ps[0]+ps[2] if x.size != 0))
    _exec_si(si, seed)
    for out in ps[0]:
      ground_truth[out] = out.buffer.as_buffer()
      del out.srcs # only schedule the LazyBuffer in this fuzz run

  # exec and validate each permutation with new Buffers
  for i, (ts, (ctx, prescheduled)) in enumerate(toposorts[1:]):
    if DEBUG >= 1: print(colored(f"testing permutation {i} {ctx}", "yellow"))
    rawbufs: Dict[LazyBuffer, Buffer] = {}
    for key in ts:
      for out in (ps:=prescheduled[key])[0]:
        rawbufs[out] = Buffer(out.buffer.device, out.buffer.size, out.buffer.dtype)
        if out.op is MetaOps.ASSIGN: rawbufs[out].ensure_allocated().copyin(prerealized[out])
      for x in ps[2]:
        if x not in rawbufs:
          # override the assign_target after ASSIGN
          if x in assign_targets and assign_targets[x] in rawbufs: rawbufs[x] = rawbufs[assign_targets[x]]
          elif x.device == "NPY": rawbufs[x] = x.buffer
          # copy the pre realized input
          else: rawbufs[x] = Buffer(x.buffer.device, x.buffer.size, x.buffer.dtype, initial_value=prerealized[x])
      si = ScheduleItem(ps[1], tuple(rawbufs[x] for x in ps[0]+ps[2] if x.size != 0))
      _exec_si(si, seed)
      for out in ps[0]:
        outbuf = np.frombuffer(rawbufs[out].as_buffer(), _to_np_dtype(out.dtype))
        try: np.testing.assert_allclose(outbuf, np.frombuffer(ground_truth[out], _to_np_dtype(out.dtype)), atol=1e-2, rtol=1e-2)
        except Exception as e:
          print(f"FAILED FOR {out}")
          raise e

def _exec_si(si:ScheduleItem, seed:int):
  ei = lower_schedule_item(si)
  if len(capturing): capturing[0].add(ei)
  if isinstance(ei.prg, CustomOp): Tensor._seed = seed
  ei.run()

T = TypeVar("T")
def find_all_toposorts(graph:DefaultDict[T, List[T]], in_degree:Union[DefaultDict[T, int], Dict[T, int]]) -> List[Tuple[T, ...]]:
  visited: Set[T] = set()
  ret: List[Tuple[T, ...]] = []
  path: List[T] = []

  def recurse_paths(path:List[T]):
    for v, d in in_degree.items():
      if d != 0 or v in visited: continue
      for u in graph[v]: in_degree[u] -= 1
      path.append(v)
      visited.add(v)
      recurse_paths(path)
      if len(ret) >= FUZZ_SCHEDULE_MAX_PATHS: return
      # backtrack
      for u in graph[v]: in_degree[u] += 1
      path.pop()
      visited.remove(v)
    if len(path) == len(in_degree): ret.append(tuple(path))
  recurse_paths(path)

  if len(ret) == 0: raise RuntimeError("detected cycle in the graph")
  # verify all paths are unique
  assert len(ret) == len(set(ret))
  return ret
