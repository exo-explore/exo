from __future__ import annotations
from typing import Optional, Tuple, Any, Set, cast, List, Union
import functools
from enum import Enum, auto
from dataclasses import dataclass
from tinygrad.dtype import ConstType, dtypes, DType
from tinygrad.shape.symbolic import sint, Variable
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, exec_alu
from tinygrad.helpers import prod

# the order of these UOps controls the order of the toposort
class UOps(Enum):
  # ops that aren't rendered
  SINK = auto(); VAR = auto(); EXPAND = auto(); CONTRACT = auto() # noqa: E702
  DEFINE_GLOBAL = auto(); DEFINE_VAR = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # noqa: E702
  CONST = auto(); SPECIAL = auto() # noqa: E702
  NOOP = auto(); UNMUL = auto(); GEP = auto() # noqa: E702
  # math ops
  CAST = auto(); BITCAST = auto(); VECTORIZE = auto() # noqa: E702
  ALU = auto(); REDUCE = auto(); WMMA = auto() # noqa: E702
  # memory/assignment ops
  LOAD = auto(); STORE = auto(); PHI = auto() # noqa: E702
  # control flow ops
  BARRIER = auto(); IF = auto(); RANGE = auto() # noqa: E702
  # these two are not graph nodes
  ENDRANGE = auto(); ENDIF = auto() # noqa: E702

END_FOR_UOP = {UOps.IF:(UOps.STORE, UOps.ENDIF), UOps.RANGE:(UOps.PHI, UOps.ENDRANGE)}

def ufix(dtype: Optional[DType], x): return UOp.const(dtype, x) if not isinstance(x, UOp) else x
@dataclass(frozen=True, eq=False)
class UOp:
  op: UOps
  dtype: Optional[DType] = None
  src: Tuple[UOp, ...] = tuple()
  arg: Any = None
  def commutative(self) -> bool:
    return self.op is UOps.ALU and \
      self.arg in {BinaryOps.ADD, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPNE, BinaryOps.XOR, BinaryOps.AND, BinaryOps.OR}
  @functools.cached_property
  def cmp_tuple(self):
    # NOTE: this sort of DEFINE_VAR shouldn't have to be here. only for PTX
    return (self.op.value, (self.arg if self.op is not UOps.DEFINE_VAR else self.arg.expr) if self.op is not UOps.ALU else \
            self.arg.value, self.dtype, self.src)
  def __lt__(self, x:UOp): return self.cmp_tuple < x.cmp_tuple
  def __repr__(self):
    return f"{str(self.op):20s}: {str(self.dtype) if self.dtype is not None else '':25s} {str([x.op for x in self.src]):32s} {self.arg}"
  def cast(self, dtype=None): return UOp(UOps.CAST, dtype, (self,))
  def bitcast(self, dtype=None): return UOp(UOps.BITCAST, dtype, (self,))
  def name(self, name:Optional[str]): return UOp(UOps.VAR, src=(self,), arg=name)
  def __neg__(self): return UOp.alu(UnaryOps.NEG, self)
  def __add__(self, x): return UOp.alu(BinaryOps.ADD, self, ufix(self.dtype, x))
  def __radd__(self, x): return UOp.alu(BinaryOps.ADD, ufix(self.dtype, x), self)
  def __sub__(self, x): return UOp.alu(BinaryOps.ADD, self, -ufix(self.dtype, x))
  def __mul__(self, x): return UOp.alu(BinaryOps.MUL, self, ufix(self.dtype, x))
  def __rmul__(self, x): return UOp.alu(BinaryOps.MUL, ufix(self.dtype, x), self)
  def __floordiv__(self, x): return UOp.alu(BinaryOps.IDIV, self, ufix(self.dtype, x))
  def __truediv__(self, x): return UOp.alu(BinaryOps.MUL, self, UOp.alu(UnaryOps.RECIP, ufix(self.dtype, x)))
  def __mod__(self, x): return UOp.alu(BinaryOps.MOD, self, ufix(self.dtype, x))
  def __xor__(self, x): return UOp.alu(BinaryOps.XOR, self, ufix(self.dtype, x))
  def __and__(self, x): return UOp.alu(BinaryOps.AND, self, ufix(self.dtype, x))
  def __or__(self, x): return UOp.alu(BinaryOps.OR, self, ufix(self.dtype, x))
  def ne(self, x): return UOp.alu(BinaryOps.CMPNE, self, ufix(self.dtype, x))
  def eq(self, x): return -self.ne(x)
  def lt(self, x): return UOp.alu(BinaryOps.CMPLT, self, ufix(self.dtype, x))
  def ge(self, x): return -self.lt(x)
  def max(self, x): return UOp.alu(BinaryOps.MAX, self, x)
  def min(self, x): return -UOp.alu(BinaryOps.MAX, -self, -x)
  def where(self, x, y): return UOp.alu(TernaryOps.WHERE, self, x, y)
  def recip(self): return UOp.alu(UnaryOps.RECIP, self)
  def const(self:Union[UOp, DType, None], b:ConstType|Variable): return UOp._const(self.dtype if isinstance(self, UOp) else self, b)
  @staticmethod
  @functools.lru_cache(maxsize=None)
  def _const(dtype:Optional[DType], b:ConstType|Variable):
    if isinstance(b, Variable): return UOp(UOps.DEFINE_VAR, dtype, (), b)
    return UOp(UOps.CONST, dtype, arg=dtypes.as_const(b, dtype) if dtype is not None else b)
  @staticmethod
  def alu(arg, *src:UOp): return UOp(UOps.ALU, dtypes.bool if arg in {BinaryOps.CMPLT, BinaryOps.CMPNE} else src[-1].dtype, src, arg)
  @staticmethod
  def load(*src:UOp, dtype:Optional[DType]=None, **kwargs): return UOp(UOps.LOAD, dtype, tuple(src)+tuple(kwargs.values()))
  @staticmethod
  def store(*src:UOp, **kwargs): return UOp(UOps.STORE, None, tuple(src)+tuple(kwargs.values()))
  @staticmethod
  def var(name:Optional[str]=None, dtype:Optional[DType]=None): return UOp(UOps.VAR, dtype=dtype, arg=name)
  @staticmethod
  def cvar(name:Optional[str]=None, dtype:Optional[DType]=None): return UOp(UOps.CONST, dtype=dtype).name(name)
  @functools.cached_property
  def parents(self) -> Set[UOp]: return set.union(set(self.src), *[x.parents for x in self.src])
  @property  # parents with self
  def sparents(self) -> Set[UOp]: return set([self]).union(self.parents)
  def vars(self) -> Set[UOp]: return set([x for x in set.union(set([self]), self.parents) if x.op is UOps.DEFINE_VAR])
  def divides(self, v):
    if self.op is UOps.CONST:
      return self.arg%v == 0
    if self.op is UOps.ALU:
      if self.arg is BinaryOps.ADD: return all(x.divides(v) for x in self.src)
      if self.arg is BinaryOps.MUL: return any(x.divides(v) for x in self.src)
    return False # generic false if we aren't sure

def type_verify(uops):
  for u in uops:
    uop, arg, src, dtype = u.op, u.arg, u.src, u.dtype
    if uop in {UOps.CONST, UOps.DEFINE_ACC}:
      if uop is UOps.DEFINE_ACC:
        assert dtype is not None and src[0].dtype == dtype.scalar(), f"type of {src[0].dtype=} must be a scalar {dtype.scalar()}"
        arg = src[0].arg
      assert dtype is not None and type(arg) is type(dtypes.as_const(arg, dtype)), f"type of {arg=} does not match {dtype}"
    if uop in {UOps.CAST, UOps.BITCAST, UOps.VECTORIZE}: assert arg is None and dtype is not None # type is the output type, not an arg
    if uop is UOps.CAST: assert dtype.count == 1 and len(src) == dtype.count
    if uop is UOps.VECTORIZE:
      assert dtype.count > 1 and len(src) == dtype.count, f"dtype vectorization mismatch {dtype.count=} != {len(src)=}"
      assert dtype == src[0].dtype.vec(len(src)), f"{dtype=} must be {src[0].dtype.vec(len(src))}"
    if uop is UOps.LOAD and len(src) > 3 and src[2].op is UOps.ALU: assert src[2].dtype == dtypes.bool and src[3].dtype == dtype
    if uop is UOps.STORE:
      assert dtype is None, f"{uop} dtype must be None, got {dtype}"
      if len(src) == 4: assert src[3].dtype == dtypes.bool, f"gate dtype mismatch {src[3].dtype} != {dtypes.bool}"
    if uop is UOps.ALU:
      if arg in UnaryOps:
        assert dtype == src[0].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=}"
      elif arg in {BinaryOps.CMPLT, BinaryOps.CMPNE}:
        assert dtype == dtypes.bool, f"{arg} output dtype mismatch {dtype=} != {dtypes.bool}"
        assert src[0].dtype == src[1].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=} != {src[1].dtype=}"
      elif arg is BinaryOps.IDIV:
        assert dtypes.is_int(src[0].dtype) and dtypes.is_int(src[1].dtype), \
            f"input dtype mismatch {dtypes.int} != {src[0].dtype=} != {src[1].dtype=}"
        assert dtypes.is_int(dtype), f"{arg} output dtype mismatch {dtype=} != {dtypes.int}"
      elif arg in {BinaryOps.SHL, BinaryOps.SHR}:
        # the distance to shift isn't typechecked
        assert dtype == src[0].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=}"
      elif arg in BinaryOps:
        assert dtype == src[0].dtype == src[1].dtype, f"{arg} dtype mismatch {dtype=} != {src[0].dtype=} != {src[1].dtype=}"
      elif arg == TernaryOps.WHERE:
        assert src[0].dtype == dtypes.bool, f"{arg} selector dtype mismatch {src[0].dtype=} != {dtypes.bool}"
        assert dtype == src[1].dtype == src[2].dtype, f"{arg} choice dtype mismatch {dtype=} != {src[1].dtype=} != {src[2].dtype=}"

def uop_alu_resolve(u:UOp) -> sint:
  if u.op is UOps.SPECIAL: return u.arg[2]-1
  if u.op in {UOps.CONST, UOps.DEFINE_VAR}: return u.arg
  if u.op is UOps.ALU: return exec_alu(u.arg, cast(DType,u.dtype), tuple(map(uop_alu_resolve, u.src)))
  raise RuntimeError(f"ALU resolve fail @ {u.op}")

def flops_mem(uops:List[UOp], ignore_indexing=False) -> Tuple[sint, sint]:
  flops: sint = 0
  mem: sint = 0
  mults: sint = 1
  mult_stack: List[sint] = []
  dont_count: Set[UOp] = set()
  if ignore_indexing:
    for u in uops:
      if u.op is UOps.LOAD:
        dont_count = dont_count.union(u.src[1].sparents)
        if len(u.src) > 3: dont_count = dont_count.union(u.src[2].sparents)
      elif u.op is UOps.STORE:
        dont_count = dont_count.union(u.src[1].sparents)
        if len(u.src) > 3: dont_count = dont_count.union(u.src[3].sparents)
  for u in uops:
    if u.op is UOps.RANGE:
      mult_stack.append(mults)
      mults *= uop_alu_resolve(u.src[1])
    elif u.op is UOps.ENDRANGE:
      mults = mult_stack.pop(-1)
    elif u.op is UOps.LOAD:
      assert u.dtype is not None
      mem += u.dtype.itemsize * mults
    elif u.op is UOps.STORE:
      assert u.src[2].dtype is not None
      mem += u.src[2].dtype.itemsize * mults
    elif u.op is UOps.ALU and u not in dont_count:
      flops += mults * (2 if u.arg == TernaryOps.MULACC else 1)
    elif u.op is UOps.WMMA and u not in dont_count:
      assert u.arg[1] is not None
      flops += 2 * prod(u.arg[1]) // 32 * mults
  return flops, mem
