from typing import Dict, List, cast, DefaultDict, Optional, Tuple, Callable
import itertools, functools, random, math, time, multiprocessing, traceback, signal
from collections import defaultdict
from dataclasses import replace
from tinygrad.device import Device, Buffer, Compiler
from tinygrad.ops import MemBuffer
from tinygrad.helpers import prod, flatten, DEBUG, CACHELEVEL, diskcache_get, diskcache_put, getenv, Context, colored, to_function_name
from tinygrad.dtype import ImageDType
from tinygrad.codegen.kernel import Kernel
from tinygrad.codegen.kernel import Opt, OptOps, KernelOptError
from tinygrad.codegen.uopgraph import UOpGraph
from tinygrad.tensor import Tensor
from tinygrad.shape.symbolic import sym_infer
from tinygrad.engine.realize import CompiledRunner
from tinygrad.renderer import Program

actions = [Opt(op=OptOps.UPCAST, axis=axis, amt=amt) for amt in [0,2,3,4,5,7] for axis in range(6)]
actions += [Opt(op=OptOps.UNROLL, axis=axis, amt=amt) for amt in [0,4,7] for axis in range(5)]
actions += [Opt(op=OptOps.LOCAL, axis=axis, amt=amt) for amt in [2,3,4,8,13,16,29] for axis in range(5)]
actions += [Opt(op=OptOps.GROUPTOP, axis=axis, amt=amt) for amt in [13,16,28,29,32,49,64,256] for axis in range(3)]
actions += [Opt(op=OptOps.GROUP, axis=axis, amt=amt) for amt in [0,4,8,16] for axis in range(3)]
if getenv("BEAM_PADTO", 1): actions += [Opt(op=OptOps.PADTO, axis=axis, amt=amt) for amt in [32] for axis in range(7)]
actions += [Opt(op=OptOps.LOCAL, axis=0, amt=32), Opt(op=OptOps.UPCASTMID, axis=1, amt=4), Opt(op=OptOps.TC, axis=0, amt=0)]
actions += [Opt(op=OptOps.TC, axis=axis, amt=getenv("TC_OPT", 2)) for axis in range(9)] # covers resnet kernels (3 global * 3 reduce)
if getenv("NOLOCALS"): actions += [Opt(op=OptOps.NOLOCALS)]

def _get_test_global_size(global_size, max_global_size, var_vals):
  test_global_size, factor = [sym_infer(sz, var_vals) for sz in global_size], 1
  while prod(test_global_size) > max_global_size:
    for j in range(len(global_size)-1,-1,-1):
      if test_global_size[j] > 16:
        test_global_size[j] //= 2
        factor *= 2
        break
  return test_global_size, factor

def _time_program(p:Program, lib:bytes, var_vals, rawbufs, early_stop=None, max_global_size=65536, clear_l2=False, cnt=3, name="test"):
  factor = 1
  if p.global_size is not None and max_global_size is not None:
    global_size, factor = _get_test_global_size(p.global_size, max_global_size, var_vals)
    p = replace(p, global_size=global_size)
  try: car = CompiledRunner(p, precompiled=lib)
  except AssertionError: return [math.inf] * cnt
  tms = []
  input_bufs = [rawbufs[i] for i,_ in car.p.globals]
  for _ in range(cnt):
    if clear_l2:
      with Context(DEBUG=0, BEAM=0, CAPTURING=0): Tensor.ones(1024,1024).contiguous().realize(do_update_stats=False)
    tms.append(cast(float, car(input_bufs, var_vals, wait=True))*factor)
    if early_stop is not None and early_stop < tms[-1]: break
  return tms

class TimeoutException(Exception): pass
def timeout_handler(signum, frame): raise TimeoutException()

def _try_compile_linearized_w_idx(x:Tuple[int,Kernel], compiler:Compiler) -> Tuple[int, Optional[Tuple[Program, bytes, float]]]:
  signal.signal(signal.SIGALRM, timeout_handler)
  # set timeout
  signal.alarm(getenv("BEAM_TIMEOUT_SEC", 10))
  try:
    x[1].linearize()
    if len(x[1].uops.uops) >= getenv("BEAM_UOPS_MAX", 3000) > 0: raise RuntimeError("too many uops")
    p = x[1].to_program()
    st = time.perf_counter()
    prog = compiler.compile(p.src)
    et = time.perf_counter() - st
    ret = (p, prog, et)
  except RuntimeError:
    if DEBUG >= 4: traceback.print_exc()
    ret = None
  except TimeoutException:
    ret = None
  except Exception as e:
    if getenv("BEAM_STRICT_MODE"): raise e
    ret = None
  finally:
    signal.alarm(0)
  return x[0], ret

# workers should ignore ctrl c
def _init_worker(): signal.signal(signal.SIGINT, signal.SIG_IGN)

def _ensure_buffer_alloc(bufs:List[Buffer]) -> List[Buffer]: return [buf.ensure_allocated() for buf in bufs]

# *** external API ***

# get (scrap) buffers for timing the linearizer
def bufs_from_lin(lin:Kernel, allocate:bool=True) -> List[Buffer]:
  bufsts:DefaultDict[int, List[MemBuffer]] = defaultdict(list)
  for x in lin.membufs: bufsts[x.idx].append(x)
  rawbufs:List[Optional[Buffer]] = [None]*len(bufsts)
  for k,lx in bufsts.items():
    buf_size = prod(lx[0].dtype.shape) if isinstance(lx[0].dtype, ImageDType) else max(y.st.real_size() for y in lx)
    if buf_size == 0: buf_size = 1  # create a size 1 buffer if no cell is accessed in kernel. # TODO: remove from kernel input in this case.
    rawbufs[k] = Buffer(lin.opts.device, buf_size, lx[0].dtype).allocate() if allocate else Buffer(lin.opts.device, buf_size, lx[0].dtype)
  assert all(r is not None for r in rawbufs)
  return cast(List[Buffer], rawbufs)

# get dictionary of all possible actions
def get_kernel_actions(lin:Kernel, include_0=True) -> Dict[int, Kernel]:
  acted_lins, max_up, max_lcl = {0:lin} if include_0 else {}, getenv("BEAM_UPCAST_MAX", 256), getenv("BEAM_LOCAL_MAX", 1024)
  for i,a in enumerate(actions):
    if a.axis is not None and a.op is not OptOps.TC:
      if ((ax:=a.real_axis(lin)) >= lin.shape_len) or (lin.full_shape[ax] == a.amt and Opt(a.op, ax, 0) in actions): continue
    lin2 = lin.copy()
    try:
      lin2.apply_opt(a)
      up, lcl, tc_up = 1, 1, prod(tc.dims)//prod([x[1] for x in tc.threads]) if (tc:=lin2.tensor_core) else 1
      for s,c in zip(lin2.full_shape, lin2.colors()):
        if c in {"magenta", "yellow"}: up *= s
        elif c in {"cyan", "green", "white"}: lcl *= s
      if up//tc_up > max_up or lcl > max_lcl: continue
      acted_lins[i+1] = lin2
    except KernelOptError: pass
  return acted_lins

beam_pool, BEAM_DEBUG = None, getenv("BEAM_DEBUG")
def beam_search(lin:Kernel, rawbufs:List[Buffer], amt:int, allow_test_size=True) -> Kernel:
  global beam_pool
  key = {"ast": lin.ast.key, "amt": amt, "allow_test_size": allow_test_size, "device": lin.opts.device, "suffix": lin.opts.suffix}
  if not getenv("IGNORE_BEAM_CACHE") and CACHELEVEL >= 1 and (val:=diskcache_get("beam_search", key)) is not None:
    ret = lin.copy()
    for o in val[len(lin.applied_opts):]: ret.apply_opt(o)
    return ret

  beam: List[Tuple[Kernel, float]] = [(lin, float("inf"))]
  seen_libs = set()

  default_parallel = multiprocessing.cpu_count() if lin.opts.device in {"CUDA", "AMD", "NV"} else 0
  if beam_pool is None and (workers := getenv("PARALLEL", default_parallel)):
    beam_pool = multiprocessing.get_context("spawn").Pool(workers, _init_worker, (), getenv("BEAM_MAX_TASKS_PER_CHILD", 16))

  min_progress = getenv("BEAM_MIN_PROGRESS", 0.01)/1e6
  if BEAM_DEBUG: print(f"BEAM_SEARCH:\n{lin.ast}")
  if DEBUG >= 2: print(f"   0.00s:                 from   1 ->   1 actions {lin.colored_shape()}")

  try:
    rawbufs = _ensure_buffer_alloc(rawbufs)
    var_vals = {k:(k.max+k.min)//2 for k in lin.ast.vars()}
    exiting, st = False, time.perf_counter()
    dev = Device[lin.opts.device]
    while not exiting:
      acted_lins: List[Kernel] = flatten([get_kernel_actions(lin, include_0=False).values() for lin,_ in beam])
      timed_lins: List[Tuple[Kernel, float]] = []
      _compile_fn = functools.partial(_try_compile_linearized_w_idx, compiler=dev.compiler)
      for i,proc in (map(_compile_fn, enumerate(acted_lins)) if beam_pool is None else beam_pool.imap_unordered(_compile_fn, enumerate(acted_lins))):
        if proc is None: continue
        p, lib, compile_et = proc
        if lib in seen_libs: continue
        #print(acted_lins[i].colored_shape(), acted_lins[i].applied_opts)  # for debugging BEAMs that segfault
        seen_libs.add(lib)
        try: tms = _time_program(p, lib, var_vals, rawbufs, early_stop=beam[0][1]*3 if len(beam) else 1.0)
        except RuntimeError: continue # for runtime issues
        timed_lins.append((acted_lins[i], min(tms)))
        if BEAM_DEBUG > 1: print(f"{time.perf_counter() - st:7.2f}s: {i:5d} {len(cast(UOpGraph, p.uops).uops):5d} uops {compile_et*1e6:12.2f} us compile/{timed_lins[-1][1]*1e6:12.2f} us run       {len(timed_lins):4d}/{len(acted_lins):4d}         {timed_lins[-1][0].colored_shape()}")  # noqa: E501
        elif DEBUG >= 2: print(f"\r{time.perf_counter() - st:7.2f}s: {timed_lins[-1][1]*1e6:12.2f} us       {len(timed_lins):4d}/{len(acted_lins):4d}         {timed_lins[-1][0].colored_shape()}\033[K", end="")  # noqa: E501

      # done
      opts = sorted(timed_lins, key=lambda x: x[1])
      exiting = len(opts) == 0 or (opts[0][1] < min_progress) or (len(beam) > 0 and ((beam[0][1]-opts[0][1]) < min_progress))
      if not exiting: beam = opts[:amt]
      elif len(opts) > 0 and opts[0][1] < beam[0][1]: beam = opts[:1]
      if DEBUG >= 2: print(f"\r{time.perf_counter() - st:7.2f}s:", colored(f"{beam[0][1]*1e6:12.2f} us", "green" if exiting else None), f"from {len(acted_lins):3d} -> {len(opts):3d} actions\033[K", beam[0][0].colored_shape())  # noqa: E501
  except KeyboardInterrupt as e:
    if beam_pool is not None: beam_pool.terminate()
    raise e

  if CACHELEVEL >= 1: diskcache_put("beam_search", key, beam[0][0].applied_opts)
  if BEAM_DEBUG: print(f"BEAM_SEARCH: final tm={beam[0][1]*1e6:0.2f} us, applied_opts={beam[0][0].applied_opts}")
  return beam[0][0]

def optimize_local_size(clprg:Callable, global_size:List[int], rawbufs:List[Buffer]) -> List[int]:
  test_rawbuffers = [Buffer(rawbufs[0].device, rawbufs[0].size, rawbufs[0].dtype).allocate(), *rawbufs[1:]] if rawbufs[0] in rawbufs[1:] else rawbufs
  MAX_WORKGROUP = 1024
  local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in global_size]
  local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
  def try_exec(local_size):
    try: return clprg(*[x._buf for x in test_rawbuffers], global_size=[g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)], local_size=local_size, wait=True)  # noqa: E501
    except Exception: return float('inf')
  ret = min([(try_exec(local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])
  assert not math.isinf(ret[0]), "all optimize_local_size exec failed"
  return ret[1]

def time_linearizer(lin:Kernel, rawbufs:List[Buffer], allow_test_size=True, max_global_size=65536, cnt=3, disable_cache=False, clear_l2=False) -> float:  # noqa: E501
  key = {"ast": lin.ast.key, "opts": str(lin.applied_opts), "allow_test_size": allow_test_size,
         "max_global_size": max_global_size, "clear_l2": clear_l2, "device": lin.opts.device, "suffix": lin.opts.suffix}
  if not disable_cache and CACHELEVEL >= 2 and (val:=diskcache_get("time_linearizer", key)) is not None: return min(val)

  dev = Device[lin.opts.device]
  assert dev.compiler is not None

  rawbufs = _ensure_buffer_alloc(rawbufs)
  var_vals = {k:(k.max+k.min)//2 for k in lin.ast.vars()}
  p = lin.to_program()
  tms = _time_program(p, dev.compiler.compile(p.src), var_vals, rawbufs,
                      max_global_size=max_global_size if allow_test_size else None, clear_l2=clear_l2, cnt=cnt, name=to_function_name(lin.name))

  if CACHELEVEL >= 2: diskcache_put("time_linearizer", key, tms)
  return min(tms)
