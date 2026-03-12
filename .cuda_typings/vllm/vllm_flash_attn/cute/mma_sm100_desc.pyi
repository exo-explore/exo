import cutlass
import cutlass.cute as cute
from enum import IntEnum

class Major(IntEnum):
    K = 0
    MN = 1

class ScaleIn(IntEnum):
    One = 0
    Neg = 1

class Saturate(IntEnum):
    False_ = 0
    True_ = 1

class CFormat(IntEnum):
    F16 = 0
    F32 = 1
    S32 = 2

class F16F32Format(IntEnum):
    F16 = 0
    BF16 = 1
    TF32 = 2

class S8Format(IntEnum):
    UINT8 = 0
    INT8 = 1

class MXF8F6F4Format(IntEnum):
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3
    E3M2 = 4
    E2M1 = 5

class MaxShift(IntEnum):
    NoShift = 0
    MaxShift8 = 1
    MaxShift16 = 2
    MaxShift32 = 3

def to_UMMA_format(cutlass_type) -> int: ...
def to_C_format(cutlass_type) -> int: ...
def make_instr_desc(
    a_type,
    b_type,
    c_type,
    M: int,
    N: int,
    a_major: Major,
    b_major: Major,
    a_neg: ScaleIn = ...,
    b_neg: ScaleIn = ...,
    c_sat: Saturate = ...,
    is_sparse: bool = False,
    max_shift: MaxShift = ...,
) -> int: ...
def mma_op_to_idesc(op: cute.nvgpu.tcgen05.mma.MmaOp): ...

class LayoutType(IntEnum):
    SWIZZLE_NONE = 0
    SWIZZLE_128B_BASE32B = 1
    SWIZZLE_128B = 2
    SWIZZLE_64B = 4
    SWIZZLE_32B = 6

def make_smem_desc_base(
    layout: cute.Layout, swizzle: cute.Swizzle, major: Major
) -> int: ...
def make_smem_desc_start_addr(start_addr: cute.Pointer) -> cutlass.Int32: ...
