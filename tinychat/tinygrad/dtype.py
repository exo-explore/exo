from typing import Final, Optional, ClassVar, Set, Tuple, Dict, Union
from dataclasses import dataclass
import functools
from tinygrad.helpers import getenv

ConstType = Union[float, int, bool]

@dataclass(frozen=True, order=True)
class DType:
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  fmt: Optional[str]
  count: int
  def __repr__(self): return f"dtypes.{'_'*(c:=self.count!=1)}{INVERSE_DTYPES_DICT[self.name if not c else self.scalar().name]}{str(self.count)*c}"
  def vec(self, sz:int):
    assert sz > 1 and self.count == 1, f"can't vectorize {self} with size {sz}"
    return DType(self.priority, self.itemsize*sz, f"{INVERSE_DTYPES_DICT[self.name]}{sz}", None, sz)
  def scalar(self): return DTYPES_DICT[self.name[:-len(str(self.count))]] if self.count > 1 else self

# dependent typing?
@dataclass(frozen=True, repr=False)
class ImageDType(DType):
  shape: Tuple[int, ...]   # arbitrary arg for the dtype, used in image for the shape
  base: DType
  def scalar(self): return self.base
  def vec(self, sz:int): return self.base.vec(sz)
  def __repr__(self): return f"dtypes.{self.name}({self.shape})"

# @dataclass(frozen=True, init=False, repr=False, eq=False)
class PtrDType(DType):
  def __init__(self, dt:DType): super().__init__(dt.priority, dt.itemsize, dt.name, dt.fmt, dt.count)
  def __repr__(self): return f"ptr.{super().__repr__()}"
  def __hash__(self): return super().__hash__()
  def __eq__(self, dt): return self.priority==dt.priority and self.itemsize==dt.itemsize and self.name==dt.name and self.count==dt.count
  def __ne__(self, dt): return not (self == dt)

class dtypes:
  @staticmethod
  def is_float(x: DType) -> bool: return x.scalar() in (dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64)
  @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool
  def is_int(x: DType) -> bool: return x.scalar() in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.bigint) or dtypes.is_unsigned(x)
  @staticmethod
  def is_unsigned(x: DType) -> bool: return x.scalar() in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  @staticmethod
  def from_py(x) -> DType:
    if x.__class__ is float: return dtypes.default_float
    if x.__class__ is int: return dtypes.default_int
    if x.__class__ is bool: return dtypes.bool
    # put this in the last is faster because there are more items than lists/tuples to check
    if x.__class__ is list or x.__class__ is tuple: return max(dtypes.from_py(xi) for xi in x) if x else dtypes.default_float
    raise RuntimeError(f"Could not infer dtype of {x} with type {type(x)}")
  @staticmethod
  def as_const(val: ConstType, dtype:DType): return int(val) if dtypes.is_int(dtype) else float(val) if dtypes.is_float(dtype) else bool(val)
  @staticmethod
  def min(dtype:DType):
    if dtypes.is_int(dtype): return 0 if dtypes.is_unsigned(dtype) else -2**(dtype.itemsize*8-1)
    return -float("inf") if dtypes.is_float(dtype) else False
  @staticmethod
  def max(dtype:DType):
    if dtypes.is_int(dtype): return (2**(dtype.itemsize*8-(0 if dtypes.is_unsigned(dtype) else 1)))-1
    return float("inf") if dtypes.is_float(dtype) else True
  @staticmethod
  def fields() -> Dict[str, DType]: return DTYPES_DICT
  bigint: Final[DType] = DType(-1, 0, "bigint", None, 1)   # arbitrary precision integer
  bool: Final[DType] = DType(0, 1, "bool", '?', 1)
  int8: Final[DType] = DType(1, 1, "char", 'b', 1)
  uint8: Final[DType] = DType(2, 1, "unsigned char", 'B', 1)
  int16: Final[DType] = DType(3, 2, "short", 'h', 1)
  uint16: Final[DType] = DType(4, 2, "unsigned short", 'H', 1)
  int32: Final[DType] = DType(5, 4, "int", 'i', 1)
  uint32: Final[DType] = DType(6, 4, "unsigned int", 'I', 1)
  int64: Final[DType] = DType(7, 8, "long", 'l', 1)
  uint64: Final[DType] = DType(8, 8, "unsigned long", 'L', 1)
  float16: Final[DType] = DType(9, 2, "half", 'e', 1)
  # bfloat16 has higher priority than float16, so least_upper_dtype(dtypes.int64, dtypes.uint64) = dtypes.float16
  bfloat16: Final[DType] = DType(10, 2, "__bf16", None, 1)
  float32: Final[DType] = DType(11, 4, "float", 'f', 1)
  float64: Final[DType] = DType(12, 8, "double", 'd', 1)

  # dtype aliases
  half = float16; float = float32; double = float64 # noqa: E702
  uchar = uint8; ushort = uint16; uint = uint32; ulong = uint64 # noqa: E702
  char = int8; short = int16; int = int32; long = int64 # noqa: E702

  # NOTE: these are image dtypes
  @staticmethod
  def imageh(shp): return ImageDType(100, 2, "imageh", 'e', 1, shape=shp, base=dtypes.float32)
  @staticmethod
  def imagef(shp): return ImageDType(100, 4, "imagef", 'f', 1, shape=shp, base=dtypes.float32)

  default_float: ClassVar[DType] = float32
  default_int: ClassVar[DType] = int32

if (env_default_float := getenv("DEFAULT_FLOAT", "")):
  dtypes.default_float = getattr(dtypes, env_default_float.lower())
  assert dtypes.is_float(dtypes.default_float), f"{env_default_float} is not a float dtype"

# https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html
# we don't support weak type and complex type
promo_lattice = { dtypes.bool: [dtypes.int8, dtypes.uint8], dtypes.int8: [dtypes.int16], dtypes.int16: [dtypes.int32], dtypes.int32: [dtypes.int64],
  dtypes.int64: [dtypes.float16, dtypes.bfloat16], dtypes.uint8: [dtypes.int16, dtypes.uint16], dtypes.uint16: [dtypes.int32, dtypes.uint32],
  dtypes.uint32: [dtypes.int64, dtypes.uint64], dtypes.uint64: [dtypes.float16, dtypes.bfloat16],
  dtypes.float16: [dtypes.float32], dtypes.bfloat16: [dtypes.float32], dtypes.float32: [dtypes.float64], }

@functools.lru_cache(None)
def _get_recursive_parents(dtype:DType) -> Set[DType]:
  return set.union(*[_get_recursive_parents(d) for d in promo_lattice[dtype]], {dtype}) if dtype != dtypes.float64 else {dtypes.float64}
@functools.lru_cache(None)
def least_upper_dtype(*ds:DType) -> DType:
  return min(set.intersection(*[_get_recursive_parents(d) for d in ds])) if not (images:=[d for d in ds if isinstance(d, ImageDType)]) else images[0]
def least_upper_float(dt:DType) -> DType: return dt if dtypes.is_float(dt) else least_upper_dtype(dt, dtypes.float32)

# HACK: staticmethods are not callable in 3.8 so we have to compare the class
DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not (k.startswith(('__', 'default', 'bigint')) or v.__class__ is staticmethod)}
INVERSE_DTYPES_DICT = {v.name:k for k,v in DTYPES_DICT.items()}
INVERSE_DTYPES_DICT['bigint'] = 'bigint'

def sum_acc_dtype(dt:DType):
  # default acc dtype for sum
  if dtypes.is_unsigned(dt): return least_upper_dtype(dt, dtypes.uint)
  if dtypes.is_int(dt) or dt == dtypes.bool: return least_upper_dtype(dt, dtypes.int)
  return least_upper_dtype(dt, dtypes.float)