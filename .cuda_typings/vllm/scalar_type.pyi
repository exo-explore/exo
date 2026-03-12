import functools
from _typeshed import Incomplete
from dataclasses import dataclass
from enum import Enum

class NanRepr(Enum):
    NONE = 0
    IEEE_754 = 1
    EXTD_RANGE_MAX_MIN = 2

@dataclass(frozen=True)
class ScalarType:
    exponent: int
    mantissa: int
    signed: bool
    bias: int
    nan_repr: NanRepr = ...
    @functools.cached_property
    def id(self) -> int: ...
    @property
    def size_bits(self) -> int: ...
    def min(self) -> int | float: ...
    def max(self) -> int | float: ...
    def is_signed(self) -> bool: ...
    def is_floating_point(self) -> bool: ...
    def is_integer(self) -> bool: ...
    def has_bias(self) -> bool: ...
    def has_infs(self) -> bool: ...
    def has_nans(self) -> bool: ...
    def is_ieee_754(self) -> bool: ...
    def __len__(self) -> int: ...
    @classmethod
    def int_(cls, size_bits: int, bias: int | None) -> ScalarType: ...
    @classmethod
    def uint(cls, size_bits: int, bias: int | None) -> ScalarType: ...
    @classmethod
    def float_IEEE754(cls, exponent: int, mantissa: int) -> ScalarType: ...
    @classmethod
    def float_(
        cls, exponent: int, mantissa: int, finite_values_only: bool, nan_repr: NanRepr
    ) -> ScalarType: ...
    @classmethod
    def from_id(cls, scalar_type_id: int): ...

class scalar_types:
    int4: Incomplete
    uint4: Incomplete
    int8: Incomplete
    uint8: Incomplete
    float8_e4m3fn: Incomplete
    float8_e5m2: Incomplete
    float8_e8m0fnu: Incomplete
    float16_e8m7: Incomplete
    float16_e5m10: Incomplete
    float6_e3m2f: Incomplete
    float6_e2m3f: Incomplete
    float4_e2m1f: Incomplete
    uint2b2: Incomplete
    uint3b4: Incomplete
    uint4b8: Incomplete
    uint8b128: Incomplete
    bfloat16 = float16_e8m7
    float16 = float16_e5m10
