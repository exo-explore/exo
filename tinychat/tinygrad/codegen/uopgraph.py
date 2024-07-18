from __future__ import annotations
from typing import Iterator, Optional, Tuple, Any, Dict, List, DefaultDict, Set, Callable, Union, cast, TYPE_CHECKING
import functools, itertools, heapq, math
from collections import defaultdict
from tinygrad.dtype import dtypes, DType, PtrDType, ImageDType
from tinygrad.shape.symbolic import Variable
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps, exec_alu
from tinygrad.helpers import DEBUG, getenv, flatten, dedup, TRANSCENDENTAL, prod, CI
from tinygrad.codegen.uops import UOp, UOps, END_FOR_UOP, type_verify
from tinygrad.codegen.transcendental import xexp2, xlog2, xsin, TRANSCENDENTAL_SUPPORTED_DTYPES

if TYPE_CHECKING:
  from tinygrad.renderer import Renderer

# *** simplification logic ***

class UPat:
  def __init__(self, op:Optional[Union[UOps, Set[UOps]]]=None, arg:Any=None, src:Optional[Union[Tuple[UPat, ...], List[UPat], UPat]]=None,
               name:Optional[str]=None, dtype:Optional[Union[DType, Set[DType]]]=None, allow_any_len:bool=False):
    self.op: Optional[Tuple[UOps, ...]] = None if op is None else (tuple(op) if isinstance(op, set) else (op,))
    self.dtype: Optional[Tuple[DType, ...]] = None if dtype is None else (tuple(dtype) if isinstance(dtype, set) else (dtype,))
    self.arg = arg
    self.src: Any = None
    if isinstance(src, list):
      # try all permutations if it's a list
      self.src = list(itertools.permutations(src))
    elif isinstance(src, tuple):
      # only one if it's a tuple
      self.src = [src]
    elif isinstance(src, UPat):
      # repeat if it's a UPat
      self.src = [itertools.repeat(src)]
    self.name: Optional[str] = name
    self.allowed_len: int = 0 if allow_any_len or isinstance(src, UPat) or src is None else len(src)

  @staticmethod
  def compile(u: UOp, name:Optional[str]=None) -> UPat:
    if u.op is UOps.VAR: return UPat(name=name or u.arg, dtype=u.dtype) if len(u.src) == 0 else UPat.compile(u.src[0], name or u.arg)
    return UPat(u.op, u.arg, (list if u.commutative() else tuple)([UPat.compile(src) for src in u.src]) if u.src != () else None,
                name, u.dtype, allow_any_len=(isinstance(name, str) and 'allow_any_len' in name))

def _match(uop:UOp, pat:UPat, store:Dict[str, UOp]) -> List[Dict[str, UOp]]:
  if (pat.name is not None and store.setdefault(pat.name, uop) is not uop) or \
     (pat.dtype is not None and uop.dtype is not None and uop.dtype not in pat.dtype) or \
     (pat.arg is not None and pat.arg != uop.arg) or \
     (pat.op is not None and uop.op not in pat.op): return []
  if pat.src is None: return [store]
  res: List[Dict[str, UOp]] = []
  for vp in pat.src:
    if pat.allowed_len != 0 and len(uop.src) != pat.allowed_len: return []
    new_stores = [store.copy()]
    for uu, vv in zip(uop.src, vp): new_stores = [rstore for nstore in new_stores for rstore in _match(uu, vv, nstore)]
    res.extend(new_stores)
  return res

class PatternMatcher:
  def __init__(self, patterns:List[Tuple[Union[UPat, UOp], Callable]]):
    self.patterns = patterns
    self.pdict: DefaultDict[Tuple[UOps, Any], List[Tuple[UPat, Callable]]] = defaultdict(list)
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      if isinstance(p, UOp): p = UPat.compile(p)
      assert p.op is not None
      for uop in p.op: self.pdict[(uop, p.arg)].append((p, fxn))

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def __add__(self, more:PatternMatcher): return PatternMatcher(self.patterns+more.patterns)

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    for p,fxn in itertools.chain(self.pdict[(uop.op, uop.arg)], self.pdict[(uop.op, None)]):
      if (matches := _match(uop, p, {})) and (ret:=fxn(**matches[0])) is not None: return ret # NOTE: if it returns None, we keep trying to match
    return None

# ***** image handling *****

def fix_image_idx(ls:UOp):
  if ls.src[1].dtype is None or ls.src[1].dtype.count != 1: return None
  if not isinstance(ls.src[0].dtype, ImageDType): return None
  assert ls.op is not UOps.STORE or cast(DType, ls.src[2].dtype).count == 4, "image store must be float4"
  idxy = ls.src[1]
  #if not idxy.divides(4): raise RuntimeError("image index must divide 4")
  base_shape = ls.src[0].dtype.shape
  idx, idy = (idxy // 4) % base_shape[1], (idxy // (4 * base_shape[1]))
  image_idx = UOp(UOps.VECTORIZE, cast(DType, idxy.dtype).vec(2), (idx, idy))
  if ls.op is UOps.LOAD and cast(DType, ls.dtype).count == 1:
    cconst = (UOp(UOps.VECTORIZE, cast(DType, ls.dtype).vec(4), src=(ls.src[3], ls.src[3], ls.src[3], ls.src[3])),) if len(ls.src) >= 3 else ()
    loaded = UOp(ls.op, cast(DType, ls.dtype).vec(4), (ls.src[0], image_idx) + ls.src[2:3] + cconst, ls.arg)
    subidx = idxy%4
    ret = UOp.const(ls.dtype, 0)
    for i in range(4): ret = UOp.alu(TernaryOps.WHERE, subidx.ne(i), ret, UOp(UOps.GEP, ls.dtype, (loaded,), i))
    return ret
  return UOp(ls.op, ls.dtype, (ls.src[0], image_idx) + ls.src[2:], ls.arg)

# ***** float4 handling *****

def float4_expand_load(load, buf, ex, idx=UOp.const(dtypes.int, 0), idx2=None):
  if len(ex.src) != 4: return None
  if tuple(x.arg for x in ex.src if x.op is UOps.CONST) != tuple(range(len(ex.src))): return None
  if buf.dtype != PtrDType(dtypes.float) and not isinstance(buf.dtype, ImageDType): return None
  if idx2 is not None: idx = idx + idx2
  if not idx.divides(len(ex.src)): return None

  if load.dtype.scalar() != load.dtype: return None  # how does this happen?
  vec_load = UOp(UOps.LOAD, load.dtype.vec(len(ex.src)), (buf, idx))
  return UOp(UOps.EXPAND, load.dtype, tuple(UOp(UOps.GEP, load.dtype, (vec_load,), i) for i in range(len(ex.src))), ex.arg)

def float4_contract_store(buf, ex, var, store_allow_any_len, idx=UOp.const(dtypes.int, 0), idx2=None, idx3=None):
  if len(ex.src) not in [2, 4]: return None
  if tuple(x.arg for x in ex.src if x.op is UOps.CONST) != tuple(range(len(ex.src))): return None
  if buf.dtype != PtrDType(dtypes.float) and not isinstance(buf.dtype, ImageDType): return None
  if idx2 is not None: idx = idx + idx2
  if idx3 is not None: idx = idx + idx3
  if not idx.divides(len(ex.src)): return None

  new_var = UOp(UOps.CONTRACT, var.dtype.vec(len(ex.src)), (var,), (ex.arg[0][0],))
  return UOp(UOps.STORE, None, (buf, idx, new_var) + store_allow_any_len.src[3:])

def no_float4_alu(alu):
  if alu.dtype.count == 1: return None
  alus = tuple(UOp(UOps.ALU, alu.dtype.scalar(),
                   tuple(UOp(UOps.GEP, s.dtype.scalar(), (s,), i) for s in alu.src), alu.arg) for i in range(alu.dtype.count))
  return UOp(UOps.VECTORIZE, alu.dtype, alus)

float4_folding = PatternMatcher([
  (UOp(UOps.STORE, dtype=dtypes.float, src=(UOp.var("buf"), UOp.var("idx")+
    (UOp(UOps.EXPAND, src=tuple(UOp.const(dtypes.int, i) for i in range(4))).name("ex")+UOp.var("idx2")), UOp.var("var"))).name("store"),
    lambda buf, store, idx, idx2, ex, var: UOp(UOps.STORE, store.dtype, (buf, idx+idx2+ex, var), store.arg)),
  # float(2,4) load
  (UOp(UOps.LOAD, dtype=dtypes.float, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex")+UOp.var("idx")+UOp.var("idx2"))).name("load"),
    float4_expand_load),
  (UOp(UOps.LOAD, dtype=dtypes.float, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex")+UOp.var("idx"))).name("load"), float4_expand_load),
  (UOp(UOps.LOAD, dtype=dtypes.float, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex"))).name("load"), float4_expand_load),
  # float(2,4) store
  # TODO: fold ADDs into one UOp and remove add chains
  (UOp(UOps.STORE, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex")+UOp.var("idx")+UOp.var("idx2")+UOp.var("idx3"), UOp.var("var"))).name("store_allow_any_len"),
    float4_contract_store),
  (UOp(UOps.STORE, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex")+UOp.var("idx")+UOp.var("idx2"), UOp.var("var"))).name("store_allow_any_len"),
    float4_contract_store),
  (UOp(UOps.STORE, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex")+UOp.var("idx"), UOp.var("var"))).name("store_allow_any_len"), float4_contract_store),
  (UOp(UOps.STORE, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex"), UOp.var("var"))).name("store_allow_any_len"), float4_contract_store),
  # no ALU on float4 (float4 constructor doesn't work in METAL/GPU)
  (UOp(UOps.ALU).name("alu"), no_float4_alu),
])

# ***** transcendental *****

transcendental_folding = PatternMatcher([
  (UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat(name="x"),), arg=UnaryOps.EXP2), xexp2),
  (UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat(name="d"),), arg=UnaryOps.LOG2), xlog2),
  (UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat(name="d"),), arg=UnaryOps.SIN), xsin),
])

# ***** threefry *****

def threefry2x32(x: UOp, seed: UOp):
  # split x into two uint32, since x in a uint64
  x0, x1 = (x & 0xffffffff).cast(dtypes.uint32), ((x // 2**32) & 0xffffffff).cast(dtypes.uint32)

  rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  ks = [0x0, (seed := seed.cast(dtypes.uint32)) ^ 0x1BD11BDA, seed]
  xr = [x0 + ks[-1], x1 + ks[0]]
  for i in range(5):
    for r in rotations[i % 2]: xr[0], xr[1] = (x0 := xr[0] + xr[1]), x0 ^ ((xr[1] * 2**r) + (xr[1] // 2**(32 - r)))
    xr = [(xr[0] + ks[i % 3]), (xr[1] + ks[(i + 1) % 3] + i + 1)]

  return xr[1].cast(dtypes.uint64) * 2**32 | xr[0].cast(dtypes.uint64)

# ***** main rewriter *****

def reduce_before_expand(reduce_allow_any_len, expand, x):
  red = UOp(UOps.REDUCE, x.dtype, (x,)+reduce_allow_any_len.src[1:], reduce_allow_any_len.arg)
  gep = tuple(UOp(UOps.GEP, reduce_allow_any_len.dtype, (red,), i) for i in range(x.dtype.count))
  return UOp(expand.op, expand.dtype, gep, expand.arg)

def sum_collapse(phi_input, loop, val1, val2):
  for v1,v2 in [(val1, val2), (val2, val1)]:
    if loop not in v1.parents:
      loop_range = loop.src[1]-loop.src[0]
      ret = v1*loop_range.cast(v1.dtype)
      return UOp(UOps.PHI, phi_input.dtype, (phi_input, v2))+ret
  return None

def loop_collapse(loop_start, loop_end, compval, idx, mval, multconst, rng):
  if getenv("DISABLE_LOOP_COLLAPSE") or not rng.arg[1]: return None  # must be a REDUCE
  if mval.arg >= 0 or loop_start.arg != 0:
    # TODO: support and test this with other mvals and loop_starts
    if DEBUG >= 1: print(f"WARNING, NOT FOLDING: mval:{mval.arg} loop_start:{loop_start.arg}")
    return None
  comprange = UOp.min(loop_end, UOp.max(UOp.alu(BinaryOps.IDIV, idx-compval-mval, mval) + (loop_end-loop_start), loop_start))
  return UOp(UOps.UNMUL, multconst.dtype, (comprange.cast(multconst.dtype) * multconst, loop_end-loop_start))

# this is symbolic 2.0
constant_folder = PatternMatcher([
  # CONTRACT before REDUCE
  (UPat(UOps.CONTRACT, name="con", src=UPat(UOps.REDUCE, name="red")),
   lambda con, red: UOp(UOps.REDUCE, con.dtype, (UOp(UOps.CONTRACT, con.dtype, red.src[0:1], con.arg),)+red.src[1:], red.arg)),
  # bigint is rewritten to int32
  (UPat({UOps.CONST, UOps.ALU, UOps.SPECIAL, UOps.RANGE, UOps.EXPAND}, dtype=dtypes.bigint, name="x"),
   lambda x: UOp(x.op, dtypes.int32, x.src, x.arg)),
  # VECTORIZE/GEP
  (UOp(UOps.GEP, src=(UOp(UOps.VECTORIZE).name("cast"),)).name("gep"), lambda gep, cast: cast.src[gep.arg]),
  *[(UOp(UOps.VECTORIZE, dtypes.float.vec(i), tuple(UOp(UOps.GEP, dtypes.float, src=(UOp.var('x'),), arg=j)
      for j in range(i))), lambda x: x) for i in [2, 4, 8]],
  # tensor core with a 0 input is acc
  (UOp(UOps.WMMA, src=(UOp.const(None, 0.0), UOp.var(), UOp.var('acc'))), lambda acc: acc),
  (UOp(UOps.WMMA, src=(UOp.var(), UOp.const(None, 0.0), UOp.var('acc'))), lambda acc: acc),
  # tensor core cleanups
  (UOp(UOps.REDUCE, src=(UOp(UOps.EXPAND, src=tuple(UOp(UOps.GEP, dtypes.float, src=(UOp.var('x'),), arg=i) for i in range(2))).name("expand"),))
   .name("reduce_allow_any_len"), reduce_before_expand),
  (UOp(UOps.REDUCE, src=(UOp(UOps.EXPAND, src=tuple(UOp(UOps.GEP, dtypes.float, src=(UOp.var('x'),), arg=i) for i in range(8))).name("expand"),))
   .name("reduce_allow_any_len"), reduce_before_expand),
  (UOp.var("add") + UOp(UOps.WMMA).name("wmma"),
    lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
  # threefry
  (UOp(UOps.ALU, dtype=dtypes.uint64, src=(UOp.var("x"), UOp.var("seed")), arg=BinaryOps.THREEFRY), threefry2x32),
  # arange loop folding (early)
  (UOp.where(UOp.alu(BinaryOps.CMPLT, UOp.alu(BinaryOps.ADD, UOp.var("idx"), UOp.alu(BinaryOps.MUL,
    UOp.cvar("mval"), UOp(UOps.RANGE, src=(UOp.var("loop_start"), UOp.var("loop_end"))).name("rng"))),
    UOp.cvar("compval")), UOp.cvar("multconst"), UOp.const(None,0)), loop_collapse),
  (UOp.where(UOp.alu(BinaryOps.CMPLT, UOp.alu(BinaryOps.ADD, UOp.var("idx"), UOp.alu(UnaryOps.NEG,
    UOp(UOps.RANGE, src=(UOp.var("loop_start"), UOp.var("loop_end"))).name("rng"))),
    UOp.cvar("compval")), UOp.cvar("multconst"), UOp.const(None, 0)),
    lambda **kwargs: loop_collapse(mval=UOp.const(dtypes.int, -1), **kwargs)),
  # sum collapse to mul (with possible GEP)
  (UPat(UOps.PHI, src=(UPat(UOps.DEFINE_ACC, name="phi_input", src=[UPat(UOps.CONST), UPat(UOps.RANGE, name="loop")]),
                       UPat(UOps.ALU, BinaryOps.ADD, src=(UPat(name="val1"), UPat(name="val2"))))), sum_collapse),
  (UPat(UOps.PHI, src=(UPat(UOps.GEP, name="phi_input", src=(UPat(UOps.DEFINE_ACC, src=[UPat(UOps.CONST), UPat(UOps.RANGE, name="loop")]),)),
                       UPat(UOps.ALU, BinaryOps.ADD, src=(UPat(name="val1"), UPat(name="val2"))))), sum_collapse),
  # deal with UNMUL
  (UPat(UOps.ALU, BinaryOps.MUL, [UPat(UOps.CONST, name="c1"), UPat(UOps.UNMUL, src=[UPat(UOps.CONST, name="c2"), UPat(name="v")])]),
   lambda c1,c2,v: v if c1.arg == c2.arg else None),
  (UOp(UOps.UNMUL, src=(UOp.const(None, 0).name('zero'), UOp.var())), lambda zero: zero),
  (UOp(UOps.UNMUL).name('unmul').cast().name('root'), lambda root,unmul: UOp(UOps.UNMUL, root.dtype, (unmul.src[0].cast(root.dtype), unmul.src[1]))),
  # indexing (with a multiply offset)!
  (UOp.var('idx').eq(UOp(UOps.RANGE).name("rng")).where(
    UOp(UOps.LOAD, src=(UOp.var("buf"), UOp.var('add')+UOp.var('mul')*UOp(UOps.RANGE).name("rng"))).name("ld"), UOp.const(None, 0.0)),
    lambda idx,rng,buf,add,mul,ld: UOp(UOps.UNMUL, ld.dtype, (UOp(ld.op, ld.dtype, (buf, add+mul*idx)), rng.src[1]-rng.src[0]))),
  # other arange folders
  (UOp.cvar("c1") - (UOp.var("x") + UOp.cvar("c2")), lambda c1, c2, x: (c1-c2)-x),  # c1 - (x + c2) -> (c1-c2) - x
  # max on special can go away (TODO: special should be variable, same thing applies)
  (UOp.max(UOp.cvar('c'), UOp(UOps.SPECIAL).name('s')), lambda c,s: c if (s.arg[2]-1) <= c.arg else None),
  (UOp.max(UOp.cvar('c'), UOp(UOps.SPECIAL).name('s')+UOp.cvar('c2')), lambda c,s,c2: (s+c2) if 0 >= c.arg else None),  # TODO: generic
  (UOp.max(UOp.cvar('c'), -(UOp(UOps.SPECIAL).name('s')+UOp.cvar('c2'))), lambda c,s,c2: -(s+c2) if -(s.arg[2]-1+c2.arg) >= c.arg else None),
  # max on range can go away (ugh: copy of SPECIAL, and with/without const)
  (UOp.max(UOp.cvar('c'), UOp(UOps.RANGE).name('s')), lambda c,s: s if s.src[0].arg >= c.arg else None),  # TODO: generic
  (UOp.max(UOp.cvar('c'), UOp(UOps.RANGE).name('s')+UOp.cvar('c2')), lambda c,s,c2: (s+c2) if s.src[0].arg >= c.arg else None),  # TODO: generic
  (UOp.max(UOp.cvar('c'), -(UOp(UOps.RANGE).name('s'))), lambda c,s: -s if -(s.src[1].arg-1) >= c.arg else None),
  (UOp.max(UOp.cvar('c'), -(UOp(UOps.RANGE).name('s')+UOp.cvar('c2'))), lambda c,s,c2: -(s+c2) if -(s.src[1].arg-1+c2.arg) >= c.arg else None),
  # const rules
  (UOp(UOps.GEP, src=(UOp.cvar("c"),)).name("root"), lambda root, c: UOp.const(root.dtype, c.arg)),
  (UPat(UOps.CAST, name="root", src=UPat(UOps.CONST, name="c")), lambda root, c: UOp.const(root.dtype, c.arg)),
  (UPat(UOps.VECTORIZE, name="root", src=UPat(UOps.CONST, name="c")), lambda root, c: UOp.const(root.dtype, c.arg)),
  # a phi on a DEFINE_ACC without loops or a CONST is a noop. this is for correctness, not just speed
  (UOp(UOps.PHI, src=(UOp(UOps.DEFINE_ACC).name("acc"), UOp.var("acc"))), lambda acc: UOp.cast(acc.src[0], acc.dtype)),
  (UOp(UOps.PHI, src=(UOp(UOps.DEFINE_ACC, src=(UOp.cvar(),)), UOp.var("x"))), lambda x: x),
  (UOp(UOps.PHI, src=(UOp.cvar(), UOp.var("x"))), lambda x: x),
  # a DEFINE_ACC without inputs is a const + GEP on a const is the const
  (UOp(UOps.DEFINE_ACC, src=(UOp.cvar(),)).name("root"), lambda root: UOp.cast(root.src[0], root.dtype)),
  (UOp(UOps.GEP, src=(UOp.cvar("x"),)).name("root"), lambda root,x: UOp.const(root.dtype, x.arg)),
  # max -2147483648
  (UOp.max(UOp.var('x'), UOp.const(dtypes.int, -2147483648)), lambda x: x),
  # bool < False is always false, True < bool is always false
  (UOp.var().lt(UOp.const(dtypes.bool, False)), lambda: UOp.const(dtypes.bool, False)),
  (UOp.const(dtypes.bool, True).lt(UOp.var()), lambda: UOp.const(dtypes.bool, False)),
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UOp.var().where(UOp.var("val"), UOp.var("val")), lambda val: val),
  (UOp.cvar('gate').where(UOp.var('c0'), UOp.var('c1')), lambda gate, c0, c1: c0 if gate.arg else c1),
  # ** constant folding **
  (UPat(UOps.ALU, name="root", src=UPat(UOps.CONST)), lambda root: UOp.const(root.dtype, exec_alu(root.arg, root.dtype, [x.arg for x in root.src]))),
  # ** self folding **
  (-(-UOp.var('x')), lambda x: x),    # -(-x) -> x
  (UOp.var('x') + 0, lambda x: x),    # x+0 -> x
  (UOp.var('x') * 1, lambda x: x),    # x*1 -> x
  (UOp.var('x') * -1, lambda x: -x),  # x*-1 -> -x
  (UOp.var('x') // UOp.var('x'), lambda x: UOp.const(x.dtype, 1)), # x//x -> 1
  (UOp.var('x') // 1, lambda x: x),   # x//1 -> x
  (UOp.var('x') // -1, lambda x: -x), # x//-1 -> -x
  (UOp.var('x') / UOp.var('x'), lambda x: UOp.const(x.dtype, 1)), # x/x -> 1
  (UOp.var('x') / UOp.cvar('c'), lambda x,c: x*exec_alu(UnaryOps.RECIP, c.dtype, [c.arg])),    # x/c -> x*(1/c)
  (UOp.var('x', dtype=dtypes.bool).max(UOp.const(dtypes.bool, False)), lambda x: x),  # max(x, False) -> x
  # ** zero folding **
  #x*0 -> 0 or 0*x -> 0
  #if x is nan or inf it should render the nan value.
  # NOTE: this can be wrong for loaded NaN
  (UOp.var('x') * 0, lambda x: UOp.const(x.dtype, float('nan') if isinstance(x.arg, float) and (math.isnan(x.arg) or math.isinf(x.arg)) else 0)),
  (UOp.var('x') - UOp.var('x'), lambda x: UOp.const(x.dtype, 0)),   # x-x -> 0
  # ** load/store folding **
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.load(UOp.var("buf"), UOp.var("idx"))), lambda buf,idx:UOp(UOps.NOOP)),
  # ** two stage add/sub folding **
  ((UOp.var('x') + UOp.cvar('c1')) + UOp.cvar('c2'), lambda x,c1,c2: x+UOp.const(x.dtype, exec_alu(BinaryOps.ADD, x.dtype, [c1.arg, c2.arg]))),
  ((UOp.var('x') - UOp.cvar('c1')) + UOp.cvar('c2'), lambda x,c1,c2: x+UOp.const(x.dtype, exec_alu(BinaryOps.ADD, x.dtype, [c2.arg, -c1.arg]))),
  # *** rules from symbolic ***
  # two stage mul, (x*c1)*c2 = x*(c1*c2)
  ((UOp.var("x") * UOp.cvar("c1")) * UOp.cvar("c2"), lambda x,c1,c2: x*UOp.const(x.dtype, exec_alu(BinaryOps.MUL, x.dtype, [c1.arg, c2.arg]))),
  # -(x+y) -> -x + -y
  #(-(UOp.var("x") + UOp.var("y")), lambda x,y: (-x)+(-y)),
  # x%1 -> 0
  (UOp.var("x") % UOp.const(None, 1), lambda x: UOp.const(x.dtype, 0)),
  # (x*c0)+(x*c1) -> x*(c0+c1)
  (UOp.var("x") * UOp.cvar("c0") + UOp.var("x") * UOp.cvar("c1"), lambda x,c0,c1: x*exec_alu(BinaryOps.ADD, x.dtype, [c0.arg, c1.arg])),
  # (x*c0)+(y*c0) -> (x+y)*c0
  #((UOp.var("x") * UOp.cvar("c0")) + (UOp.var("y") * UOp.cvar("c0")), lambda x,y,c0: c0*(x+y)),
  # (x*c0)//c0 -> x
  ((UOp.var("x") * UOp.cvar("c0")) // UOp.cvar("c0"), lambda x,c0: x if c0.arg != 0 else None),
  # (x*x2)/x2 -> x
  ((UOp.var("x") * UOp.var("x2")) / UOp.var("x2"), lambda x,x2: x),
  # (x//c0)//c1 -> x//(c0*c1)
  ((UOp.var("x") // UOp.cvar("c0")) // UOp.cvar("c1"), lambda x,c0,c1: x//UOp.const(x.dtype, exec_alu(BinaryOps.MUL, x.dtype, [c0.arg, c1.arg]))),
  # (x/x1)/x2 -> x/(x1*x2)
  ((UOp.var("x") / UOp.var("x2")) / UOp.var("x3"), lambda x,x2,x3: x/(x2*x3)),
  # c0 + x < c1 -> x < c1 - c0
  ((UOp.cvar("c0") + UOp.var("x")).lt(UOp.cvar("c1")),
    lambda x,c0,c1: UOp.lt(x, UOp.const(x.dtype, exec_alu(BinaryOps.ADD, x.dtype, [c1.arg, -c0.arg])))),
  # (x+x*c0)-> x*(c0+1)
  (UOp.var("x") + UOp.var("x") * UOp.cvar("c0"), lambda x,c0: x*UOp.const(x.dtype, c0.arg+1)),
  # x!=0 -> (bool)x
  (UOp.var("x").ne(0), lambda x: x.cast(dtypes.bool)),
  # bool != 1 -> not bool
  (UOp.var("x", dtype=dtypes.bool).ne(1), lambda x: -x),
  # TODO: can do the invert of this (flip alt/load) when we fix double ops
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.alu(TernaryOps.WHERE, UOp.var("gate"), UOp.var("alt"), UOp.load(UOp.var("buf"), UOp.var("idx")))),
   lambda buf, idx, gate, alt: UOp.store(buf, idx, alt, gate)),
  # VECTORIZE-PHI-GEP -> PHI-VECTORIZE
  (UOp(UOps.VECTORIZE, src=tuple(UOp(UOps.PHI, src=(UOp(UOps.GEP, src=(UOp.var("val"),), arg=i), UOp.var(f"v{i}"))) for i in range(4))).name("root"),
   lambda root, val, v0, v1, v2, v3: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.VECTORIZE, val.dtype, (v0, v1, v2, v3))))),
  (UOp(UOps.VECTORIZE, src=tuple(UOp(UOps.PHI, src=(UOp(UOps.GEP, src=(UOp.var("val"),), arg=i), UOp.var(f"v{i}"))) for i in range(2))).name("root"),
   lambda root, val, v0, v1: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.VECTORIZE, val.dtype, (v0, v1))))),
  # NEG/CMPLT -> CMPLT
  (UOp.lt(-UOp.var('x'), UOp.cvar('c', dtypes.int)), lambda c,x: UOp.lt(UOp.const(c.dtype, -c.arg), x)),
  # cast NOOP (NOTE: it's str to deal with PtrDType)
  (UOp(UOps.CAST).name("root"), lambda root: root.src[0] if str(root.dtype) == str(root.src[0].dtype) else None),
  (UOp(UOps.VECTORIZE).name("root"), lambda root: root.src[0] if str(root.dtype) == str(root.src[0].dtype) else None),
  # fold gated LOAD/STORE
  (UOp.load(UOp.var("buf"), UOp.var("idx"), UOp.const(dtypes.bool, True), UOp.cvar("var")), lambda buf,idx,var: UOp.load(buf, idx, dtype=var.dtype)),
  (UOp.load(UOp.var("buf"), UOp.var("idx"), UOp.const(dtypes.bool, True), UOp.cvar("var"), UOp.var("barrier")),
   lambda buf,idx,var,barrier: UOp.load(buf, idx, barrier, dtype=var.dtype)),
  (UOp.load(UOp.var(), UOp.var(), UOp.const(dtypes.bool, False), UOp.cvar("var")), lambda var: var),
  (UOp.load(UOp.var(), UOp.var(), UOp.const(dtypes.bool, False), UOp.cvar("var"), UOp.var()), lambda var: var),
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.var("val"), UOp.const(dtypes.bool, True)), UOp.store),
  (UOp.store(UOp.var(), UOp.var(), UOp.var(), UOp.const(dtypes.bool, False)), lambda: UOp(UOps.NOOP)),
  # remove NOOPs from SINK
  (UOp(UOps.SINK).name("root"),
    lambda root: UOp(UOps.SINK, root.dtype, a, root.arg) if len(a:=tuple(x for x in root.src if x.op is not UOps.NOOP)) != len(root.src) else None),
])

# *** uop expander ***

def _expand_arg_to_idx(args:Tuple[Tuple[int, int], ...], rpk:Dict[int, int]) -> int:
  idx, mul = 0, 1
  for axis,m in args[::-1]:
    idx += rpk[axis] * mul
    mul *= m
  return idx

def _choices_from_args(args:Tuple[Tuple[int, int], ...]) -> List[Dict[int, int]]:
  return [dict(x) for x in itertools.product(*[zip(itertools.repeat(axis), range(m)) for axis,m in args])]

def do_expand(root:UOp):
  if root.op is UOps.REDUCE:
    if root.src[0].op is not UOps.EXPAND: return None
    reduce_expand_args = flatten([x.arg for x in root.src[1:] if x.op is UOps.EXPAND])
    expand_args = tuple(x for x in root.src[0].arg if x not in reduce_expand_args)
    if len(expand_args) == 0: return None
    dont_expand_args = tuple(x for x in root.src[0].arg if x in reduce_expand_args)
  else:
    expands = [x for x in root.src if x.op is UOps.EXPAND]
    if len(expands) == 0: return None
    expand_args = tuple(sorted(dedup(flatten([x.arg for x in expands]))))
    if root.op is UOps.WMMA:
      dont_expand_args = tuple(x for x in expand_args if x[0] in root.arg[-1] or x[0] in root.arg[-2])
      expand_args = tuple(x for x in expand_args if x not in dont_expand_args)
    else:
      dont_expand_args = ()
  new_srcs: List[UOp] = []
  lrpks = _choices_from_args(dont_expand_args)
  for rpk in _choices_from_args(expand_args):
    new_src: List[UOp] = []
    for src in root.src:
      if src.op is UOps.EXPAND:
        lnew_src = [src.src[_expand_arg_to_idx(src.arg, {**rpk, **lrpk})] for lrpk in lrpks]
        if len(dont_expand_args):
          # TODO: is this right for UOps.WMMA? all lnew_src should be the same
          new_src.append(lnew_src[0] if root.op is UOps.WMMA else UOp(UOps.EXPAND, root.dtype, tuple(lnew_src), dont_expand_args))
        else:
          assert len(lnew_src) == 1
          new_src.append(lnew_src[0])
      else:
        new_src.append(src)
    new_srcs.append(UOp(root.op, root.dtype, tuple(new_src), root.arg))
  if root.op is UOps.EXPAND:
    expand_args, old_args = tuple(sorted(root.arg+expand_args)), expand_args
    assert len(expand_args) == (len(old_args) + len(root.arg))
    new_srcs = [new_srcs[_expand_arg_to_idx(old_args, rpk)].src[_expand_arg_to_idx(root.arg, rpk)] for rpk in _choices_from_args(expand_args)]
  assert prod([x[1] for x in expand_args]) == len(new_srcs)
  return UOp(UOps.EXPAND, root.dtype, tuple(new_srcs), expand_args)

acc_number = 0
def do_reduce_with_expand(root):
  global acc_number
  expands = [x for x in root.src[1:] if x.op is UOps.EXPAND]
  expands_reduce = [x for x in expands if root.src[0].op is UOps.EXPAND and all(y in root.src[0].arg for y in x.arg)]
  expands_non_reduce = [x for x in expands if x not in expands_reduce]
  const = UOp.const(root.dtype.scalar(), dtypes.as_const(0, root.dtype.scalar()) if root.arg is ReduceOps.SUM else dtypes.min(root.dtype.scalar()))
  ret = acc = UOp(UOps.DEFINE_ACC, root.dtype, (const,) + tuple(x for x in root.src[1:] if x.op is not UOps.EXPAND), (acc_number,))
  acc_number += 1
  if len(expands_reduce):
    assert root.src[0].op is UOps.EXPAND
    expand_reduce_args = dedup(flatten([x.arg for x in expands_reduce]))
    assert prod([y[1] for y in expand_reduce_args]) == len(root.src[0].src)
    for xx in root.src[0].src:
      ret = UOp.alu({ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX}[cast(ReduceOps, root.arg)], ret, xx)
  else:
    ret = UOp.alu({ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX}[cast(ReduceOps, root.arg)], ret, root.src[0])
  ret = UOp(UOps.PHI, ret.dtype, (acc, ret))
  if len(expands_non_reduce): ret = ret * prod([sz for _,sz in flatten([x.arg for x in expands_non_reduce])])
  return ret

def do_contract(con:UOp):
  ex = con.src[0]
  assert con.dtype is not None
  # CONTRACT without EXPAND repeats the element VECTORIZED
  if ex.op is not UOps.EXPAND: return UOp(UOps.VECTORIZE, con.dtype, con.src*con.dtype.count)
  # simple CONTRACT and EXPAND cancel out
  if len(ex.arg) == 1 and len(con.arg) == 1 and ex.arg[0][0] in con.arg: return UOp(UOps.VECTORIZE, con.dtype, ex.src)
  # complex CONTRACT may only remove one axis from EXPAND
  assert len(con.arg) == 1, "contract arg one is all that's supported"
  try:
    split_index = [x[0] for x in ex.arg].index(con.arg[0])
  except ValueError:
    # CONTRACT without EXPAND (still) repeats the element VECTORIZED
    return UOp(UOps.VECTORIZE, con.dtype, con.src*con.dtype.count)
  assert con.dtype.count == ex.arg[split_index][1], "contract arg must match"
  number_after = prod([x[1] for x in ex.arg[split_index+1:]])
  to_join = [ex.src[i:i+number_after] for i in range(0, len(ex.src), number_after)]
  srcs = []
  for i in range(0, len(to_join), con.dtype.count):
    srcs += [UOp(UOps.VECTORIZE, con.dtype, tuple(src)) for src in zip(*to_join[i:i+con.dtype.count])]
  return UOp(UOps.EXPAND, con.dtype, tuple(srcs), tuple(x for x in ex.arg if x[0] != con.arg[0]))

expander = PatternMatcher([
  (UPat({UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.GEP, UOps.WMMA, UOps.LOAD, UOps.STORE,
         UOps.VECTORIZE, UOps.REDUCE, UOps.EXPAND, UOps.IF}, name="root"), do_expand),
  (UOp(UOps.REDUCE).name("root"), do_reduce_with_expand),
  (UOp(UOps.CONTRACT).name("con"), do_contract),
  # remove EXPANDs from SINK
  (UOp(UOps.SINK).name("root"),
   lambda root: UOp(UOps.SINK, root.dtype, a, root.arg)
    if len(a:=tuple(flatten(x.src if x.op is UOps.EXPAND else (x,) for x in root.src))) != len(root.src) else None),
  # BARRIERs aren't actually expanded
  (UOp(UOps.BARRIER, src=(UOp(UOps.EXPAND).name("ex"),)), lambda ex: UOp(UOps.EXPAND, None, (UOp(UOps.BARRIER, None, ex.src),)*len(ex.src), ex.arg)),
  # image indexing (needs to be here)
  (UPat({UOps.LOAD, UOps.STORE}, name="ls"), fix_image_idx),
  # empty EXPAND is NOOP
  (UOp(UOps.EXPAND, src=(UOp.var('x'),), arg=()), lambda x: x),
])

# *** uop graph ***

def get_children_dfs(u:UOp, children:Dict[UOp, List[UOp]], in_degree:Dict[UOp, int]):
  if u in children: return
  children[u] = []
  for x in u.src:
    get_children_dfs(x, children, in_degree)
    children[x].append(u)
  in_degree[u] = len(u.src)

def graph_rewrite(sink:UOp, pm:PatternMatcher) -> UOp:
  nodes: Dict[Tuple, UOp] = {}
  replace: Dict[UOp, UOp] = {}
  def __inner_rewrite(n:UOp) -> UOp:
    if n in replace: return replace[n]
    replace_source = (n.op, n.dtype, tuple(__inner_rewrite(y) for y in n.src), n.arg)
    if found := nodes.get(replace_source): replace[n] = found
    else: nodes[replace_source] = replace[n] = found = __inner_rewrite(new_x) if (new_x := pm.rewrite(x:=UOp(*replace_source))) else x
    return found
  return __inner_rewrite(sink)

class UOpGraph:
  def __init__(self, sink:Union[UOp, List[UOp]], opts:Optional[Renderer]=None):
    self.sink: UOp = sink if isinstance(sink, UOp) else UOp(UOps.SINK, None, tuple(sink))
    assert self.sink.op is UOps.SINK, f"sink isn't sink, it's {self.sink.op}"
    # used by linearizer
    self._uops: Optional[List[UOp]] = None
    self.opts = opts
    self.folder = constant_folder if opts is None or not opts.supports_float4 else (constant_folder+float4_folding)
    if TRANSCENDENTAL >= 2 or (opts is not None and TRANSCENDENTAL >= 1 and opts.device in {"CLANG", "LLVM"}):
      self.folder = self.folder + transcendental_folding

  def __reduce__(self): return self.__class__, (self.sink, self.opts)
  def __iter__(self) -> Iterator[UOp]: return iter(self.uops)
  def __getitem__(self, index) -> UOp: return self.uops[index]

  def vars(self) -> List[Variable]: return sorted([x.arg for x in self.uops if x.op is UOps.DEFINE_VAR], key=lambda v: v.expr)
  def globals(self) -> List[Tuple[int, bool]]: return [x.arg for x in self.uops if x.op is UOps.DEFINE_GLOBAL]

  @property
  def uops(self) -> List[UOp]:
    if self._uops is None: self.linearize()
    return cast(List[UOp], self._uops)

  def graph(self):
    from tinygrad.engine.graph import graph_uops
    graph_uops(self.uops)

  def print(self):
    for i,u in enumerate(self):
      formatted_parents = [self.uops.index(x) if x.op is not UOps.CONST else f"{x.arg}" for x in u.src]
      print(f"{i:4d} {str(u.op):20s}: {str(u.dtype) if u.dtype is not None else '':25s} " f"{str(formatted_parents):32s} {u.arg}")

  cnt = 0
  def linearize(self, extra_pm:Optional[PatternMatcher]=None):
    global acc_number
    acc_number = 0

    # NOTE: relinearizering should be okay
    #assert self._uops is None, "already linearized"

    # fixup gated stores with an IF block to save extra local loads
    @functools.lru_cache(None)
    def _dfs(u:UOp, gate:UOp) -> UOp:
      if u.op is UOps.LOAD and u.src[-1].op is UOps.BARRIER:
        if_uop = UOp(UOps.IF, None, (gate, u.src[-1]))
        return UOp(u.op, u.dtype, u.src[:-1]+(if_uop,), u.arg)
      if (replace_source:=tuple(_dfs(x, gate) for x in u.src)) != u.src: return UOp(u.op, u.dtype, replace_source, u.arg)
      return u
    sink_srcs = list(self.sink.src)
    for i, s in enumerate(sink_srcs):
      # breaks for WMMA
      if all(x.op is not UOps.WMMA for x in s.parents):
        if s.op is UOps.STORE and len(s.src) == 4 and (rw:=_dfs(s, s.src[3])) != s:
          sink_srcs[i] = UOp(rw.op, rw.dtype, rw.src[:3], rw.arg)
    sink = UOp(UOps.SINK, None, tuple(sink_srcs))

    # do graph rewrite
    sink = graph_rewrite(sink, self.folder)

    # expand
    UOpGraph.cnt += 1
    if UOpGraph.cnt != getenv("DEBUG_EXPAND", 0): sink = graph_rewrite(sink, expander+self.folder)

    # for PTX only
    if extra_pm: sink = graph_rewrite(sink, self.folder+extra_pm)

    # filter nodes that don't link to a sink
    # BFS toposort
    children: Dict[UOp, List[UOp]] = {}
    in_degree: Dict[UOp, int] = {}
    get_children_dfs(sink, children, in_degree)

    @functools.lru_cache(None)
    def get_recursive_children(x:UOp, end:UOps, include_self=False) -> Set[UOp]:
      if x.op is UOps.SINK: return set()
      return set.union(set((x,)) if include_self else set(), *([get_recursive_children(u, end, True) for u in children[x] if x.op is not end]))

    # scope children impact the toposort and END* insertion
    scope_children = {p:get_recursive_children(p, END_FOR_UOP[p.op][0]) for p in reversed(in_degree) if p.op in END_FOR_UOP}

    queue:List[Tuple[int, UOp]] = []
    def push(u:UOp):
      priority = 0
      # prefer uops that are loop children
      for l, ss in scope_children.items():
        if l.op is UOps.RANGE and u in ss: priority -= l.arg[0]*1000 + l.arg[1]
      heapq.heappush(queue, (priority, u))

    for u in children:
      if in_degree[u] == 0: push(u)

    scope_end: Dict[UOp, UOp] = {}
    self._uops = []
    while queue:
      p,x = heapq.heappop(queue)
      if DEBUG >= 7: print(p,x)
      if x in scope_children: scope_end[x] = x
      if x.op is UOps.DEFINE_ACC:
        idx = min([self._uops.index(l) for l in x.src if l.op is UOps.RANGE])
        self._uops.insert(idx, x)
      else: self._uops.append(x)
      for u, ss in scope_children.items():
        if x in ss:
          ss.remove(x)
          if len(ss) == 0: scope_end[u] = x
      for u in children[x]:
        in_degree[u] -= 1
        if in_degree[u] == 0: push(u)

    # end scopes in toposort order
    for u, x in scope_end.items(): self._uops.insert(self._uops.index(x)+1, UOp(END_FOR_UOP[u.op][1], None, (u,)))

    # sanity checks (NOTE: these can cause things to be skipped in BEAM)
    bad_ops = dedup([x.op for x in self._uops if x.op in {UOps.EXPAND, UOps.CONTRACT, UOps.REDUCE, UOps.UNMUL}])
    try:
      type_verify(self.uops)
      assert self._uops[-1].op is UOps.SINK, f"didn't end with SINK, ended with {self._uops[-1]}"
      assert len(bad_ops) == 0, f"bad UOps left in list: {bad_ops}"
      # TODO: this should be enabled, and the valid clause should be removed
      # NOTE: multiple identical stores to DEFINE_LOCAL is okay
      assert len(all_stores := [x.src[0:2]+x.src[3:] for x in self._uops if x.op is UOps.STORE and x.src[0].op is not UOps.DEFINE_LOCAL]) \
        == len(dedup(all_stores)), "repeated stores in uops"
    except AssertionError as e:
      self.print()
      if not CI: self.graph()
      raise e

    # strip the SINK
    self._uops = self._uops[:-1]

    if getenv("FUZZ_UOPS"):
      from test.external.fuzz_uops import fuzz_uops
      self._fuzz_paths = fuzz_uops(self)
