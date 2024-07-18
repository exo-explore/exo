# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Tuple, List, Optional, Any, Dict
import pickle, base64, itertools, time, struct
from tinygrad.dtype import DType, dtypes, ImageDType
from tinygrad.helpers import all_same, getenv, flatten
from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.codegen.uops import UOps
from tinygrad.codegen.uopgraph import UOpGraph
from tinygrad.ops import BinaryOps, TernaryOps, exec_alu, truncate
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer, MetalRenderer, AMDRenderer

def _load(m, i):
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return m[i]

def load(inp, j=0):
  if len(inp) == 4: return [_load(m, x+j) if gate else default for m,x,gate,default in zip(*inp)]
  return [_load(m, x+j) for m,x in zip(inp[0], inp[1])]

def _store(m, i, v):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = v

class PythonProgram:
  def __init__(self, name:str, lib:bytes):
    self.uops: List[Tuple[UOps, Optional[DType], List[int], Any]] = pickle.loads(lib)
  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      ul: Dict[int, Any] = {}
      dl: Dict[int, DType] = {}
      pbufs: List[memoryview] = list(bufs)
      pvals: List[int] = list(vals)
      i = 0
      loop_ends: Dict[int, int] = {}
      while i < len(self.uops):
        uop, dtype, idp, arg = self.uops[i]
        void_ops = {UOps.STORE, UOps.ENDRANGE, UOps.BARRIER, UOps.IF, UOps.ENDIF}
        if uop is UOps.DEFINE_ACC: idp = [idp[0]]
        inp = [ul[v] for v in idp if self.uops[v][0] not in void_ops]
        dtp = [dl[v] for v in idp if self.uops[v][0] not in void_ops]
        if getenv("TRACE"): print(i, uop, dtype, arg, inp, dtp)
        if uop is UOps.STORE:
          if len(inp) == 3: inp.append([True] * len(inp[0]))  # set the gate to True
          if isinstance(dtp[0], ImageDType):
            # image store
            assert dtp[2].count == 4
            for j,val in enumerate(inp[2]):
              for m,ox,oy,v,g in zip(inp[0], inp[1][0], inp[1][1], val, inp[3]):
                assert ox >= 0 and ox < dtp[0].shape[1] and oy >= 0 and oy < dtp[0].shape[0]
                if g: _store(m, ox*4 + oy*dtp[0].shape[1]*4 + j, v)
          elif dtp[2].count > 1:
            for j,val in enumerate(inp[2]):
              for m,o,v,g in zip(inp[0], inp[1], val, inp[3]):
                if g: _store(m, o+j, v)
          else:
            for m,o,v,g in zip(*inp):
              if g: _store(m, o, v)
          i += 1
          continue
        if uop is UOps.ENDRANGE:
          loop_ends[idp[0]] = i
          i = idp[0]
          continue
        if uop in (UOps.BARRIER, UOps.IF, UOps.ENDIF):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        assert dtype is not None, f"{uop} is missing a dtype"
        dl[i] = dtype
        if uop is UOps.DEFINE_GLOBAL:
          assert dtype.fmt is not None
          ul[i] = [pbufs.pop(0).cast(dtype.fmt)] * warp_size
        elif uop is UOps.DEFINE_LOCAL:
          assert dtype.fmt is not None
          lbuf = memoryview(bytearray(arg[1]*dtype.itemsize))
          ul[i] = [lbuf.cast(dtype.fmt)] * warp_size
        elif uop is UOps.DEFINE_VAR:
          ul[i] = [pvals.pop(0)] * warp_size
        elif uop is UOps.SPECIAL:
          if arg[1][0] == 'g':
            ul[i] = [idxs[2-arg[0]]] * warp_size
          elif arg[1][0] == 'l':
            ul[i] = [x[2-arg[0]] for x in warp]
        elif uop is UOps.CONST:
          ul[i] = [[arg] * warp_size for _ in range(dtype.count)] if dtype.count > 1 else [arg] * warp_size
        elif uop is UOps.DEFINE_ACC:
          ul[i] = [[inp[0][0]] * warp_size for _ in range(dtype.count)] if dtype.count > 1 else [inp[0][0]] * warp_size
        elif uop is UOps.RANGE:
          if i not in ul: ul[i] = [inp[0][0]] * warp_size
          else:
            for j in range(len(ul[i])):
              ul[i][j] += 1
            if ul[i][0] == inp[1][0]:
              del ul[i]
              i = loop_ends[i] + 1
              continue
        elif uop is UOps.VECTORIZE: ul[i] = inp
        elif uop in {UOps.CAST, UOps.BITCAST}:
          assert dtp[0].fmt and dtype.fmt
          pack_format, unpack_format = str(warp_size) + dtp[0].fmt, str(warp_size) + dtype.fmt
          if uop is UOps.BITCAST: ul[i] = list(struct.unpack(unpack_format, struct.pack(pack_format, *inp[0])))
          else:
            casted = [dtypes.as_const(x, dtype) for x in inp[0]]
            if dtypes.is_int(dtype):
              overflow_adjust = 2**(dtype.itemsize*8 - 1) if not dtypes.is_unsigned(dtype) else 0
              casted = [((x + overflow_adjust) % 2**(dtype.itemsize*8) - overflow_adjust) for x in casted]
            elif dtypes.is_float(dtype):
              casted = [truncate.get(dtype, lambda dt: dt)(x) for x in casted]
            ul[i] = list(struct.unpack(unpack_format, struct.pack(unpack_format, *casted)))
        elif uop is UOps.LOAD:
          if isinstance(dtp[0], ImageDType):
            assert dtype.count == 4
            ul[i] = []
            for j in range(dtype.count):
              ret = []
              for m,ox,oy in zip(inp[0], inp[1][0], inp[1][1]):
                if ox < 0 or ox >= dtp[0].shape[1] or oy < 0 or oy >= dtp[0].shape[0]: ret.append(0)
                else: ret.append(_load(m, ox*4 + oy*dtp[0].shape[1]*4 + j))
              ul[i].append(ret)
          elif dtype.count > 1:
            ul[i] = [load([inp[i][j] if dtp[i].count > 1 else inp[i] for i in range(len(inp))], j) for j in range(dtype.count)]
          else:
            ul[i] = load(inp)
        elif uop is UOps.PHI:
          for j in range(len(inp[0])): inp[0][j] = inp[1][j]
          ul[i] = inp[0]
        elif uop is UOps.GEP:
          ul[i] = inp[0][arg]
        elif uop is UOps.WMMA:
          # here are the models for the WMMA instruction on the different hardware
          def wmma_helper(WARP_THREADS, K, NUM_A, NUM_B, NUM_C, a_elem, b_elem, c_map):
            assert len(inp[0]) == NUM_A, f"A must have {NUM_A} elements per thread, it has {len(inp[0])}"
            assert len(inp[1]) == NUM_B, f"B must have {NUM_B} elements per thread, it has {len(inp[1])}"
            assert len(inp[2]) == NUM_C, f"C must have {NUM_C} elements per thread, it has {len(inp[2])}"
            assert len(flatten(inp[0])) == NUM_A * warp_size, f"WMMA must have {NUM_A * warp_size} total elements for A in WMMA"
            assert len(flatten(inp[1])) == NUM_B * warp_size, f"WMMA must have {NUM_B * warp_size} total elements for B in WMMA"
            assert len(flatten(inp[2])) == NUM_C * warp_size, f"WMMA must have {NUM_C * warp_size} total elements for C in WMMA"
            assert warp_size > 0 and warp_size % WARP_THREADS == 0, f"must have multiples of {WARP_THREADS} warp threads"
            out = [inp[2][elem_idx][:] for elem_idx in range(NUM_C)]
            for goff in range(0, warp_size, WARP_THREADS):
              for lane_id in range(WARP_THREADS):
                for elem_idx in range(NUM_C): # calculate new muls and add to acc
                  (c_i, c_j) = c_map(lane_id, elem_idx)
                  out[elem_idx][goff+lane_id] += sum(a_elem(inp[0], _k, c_j, goff) * b_elem(inp[1], c_i, _k, goff) for _k in range(K))
            return out

          # TODO: refactor these to a shared TensorCoreLayout in kernel.py
          if arg[5] == "METAL":
            # A (2 elements on 32 threads): row major
            def a_b_elem(x, i, j, goff): return x[(i%2)][goff+(i//2)%2+(j%4)*2+(i//4)*8+(j//4)*16]
            # (i, j), C, D (2 elements on 32 threads): row major same as A/B
            def c_map(lane, elem): return (elem + ((lane%2)*2) + ((lane//8)%2)*4, ((lane//2)%4) + (lane//16)*4)
            ul[i] = wmma_helper(32, 8, 2, 2, 2, a_b_elem, a_b_elem, c_map)
          elif arg[5] == "AMD":
            # A (16 elements on 32 threads): col major, lane 16-32 == lane 0-15
            def a_elem(x, i, j, goff):
              assert x[i][goff+j] == x[i][goff+j+16], "warp elements not duplicated properly across lanes"
              return x[i][goff+j]
            # B (16 elements on 32 threads): row major, lane 16-32 == lane 0-15
            def b_elem(x, i, j, goff): return a_elem(x, j, i, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, lane//16+elem*2) # (i, j), C, D (8 elements on 32 threads): row major
            ul[i] = wmma_helper(32, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif arg[5] == "CUDA":
            # A (8 elements on 32 threads)
            def a_elem(x, i, j, goff): return x[(i%2)+(j//8)*2+(i//8)*4][goff+((i//2)%4)+(j%8)*4]
            # B (4 elements on 32 threads)
            def b_elem(x, i, j, goff): return x[(j%2)+(j//8)*2][goff+(j//2)%4+(i)*4]
            # (i, j), C, D (4 elements on 32 threads)
            def c_map(lane, elem): return ((elem%2)+(lane%4)*2, (lane//4)+(elem//2)*8)
            ul[i] = wmma_helper(32, 16, 8, 4, 4, a_elem, b_elem, c_map)
          else: raise NotImplementedError(f"unimplemented tensor core {arg}")
        elif uop is UOps.ALU:
          assert all_same([len(x) for x in inp]), f"{[len(x) for x in inp]} doesn't match on {arg}"
          assert all_same([dtype] + dtp) or arg in {BinaryOps.CMPNE, BinaryOps.CMPLT, TernaryOps.WHERE}, f"dtype mismatch on {arg}"
          ul[i] = [exec_alu(arg, dtype, p) for p in zip(*inp)]
        assert i in ul, (uop, dtype, idp, arg)
        i += 1
    return time.perf_counter() - st

class PythonRenderer(Renderer):
  device = "PYTHON"
  def __init__(self):
    if getenv("EMULATE_METAL"): self.device, self.tensor_cores = "METAL", MetalRenderer.tensor_cores
    if getenv("EMULATE_AMD"): self.device, self.tensor_cores = "AMD", AMDRenderer.tensor_cores
    if getenv("EMULATE_CUDA"): self.device, self.tensor_cores = "CUDA", CUDARenderer.tensor_cores

  def render(self, name:str, uops:UOpGraph) -> str:
    lops = [(u.op, u.dtype, [uops.uops.index(v) for v in u.src], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()

class PythonCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonAllocator(Allocator):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def copyin(self, dest, src:memoryview): dest[:] = src
  def copyout(self, dest:memoryview, src): dest[:] = src

class PythonDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, PythonAllocator(), PythonRenderer(), PythonCompiler(), PythonProgram)
