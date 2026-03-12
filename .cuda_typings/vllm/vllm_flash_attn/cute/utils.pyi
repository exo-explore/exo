import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cutlass_dsl import dsl_user_op
from typing import Callable, overload

def hash_callable(func: Callable, set_cute_hash: bool = True) -> str: ...
def create_softcap_scoremod(softcap_val): ...
def convert_from_dlpack(
    x, leading_dim, alignment: int = 16, divisibility: int = 1
) -> cute.Tensor: ...
def convert_from_dlpack_leading_static(
    x, leading_dim, alignment: int = 16, static_modes=None, stride_order=None
) -> cute.Tensor: ...
def make_tiled_copy_A(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy: ...
def make_tiled_copy_B(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy: ...
def mma_make_fragment_A(
    smem: cute.Tensor,
    thr_mma: cute.core.ThrMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.Tensor: ...
def mma_make_fragment_B(
    smem: cute.Tensor,
    thr_mma: cute.core.ThrMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.Tensor: ...
def get_smem_store_atom(
    arch: cutlass.Constexpr[int],
    element_type: type[cute.Numeric],
    transpose: bool = False,
) -> cute.CopyAtom: ...
@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = ...,
) -> cute.TensorSSA | cute.Numeric: ...
def parse_swizzle_from_pointer(ptr: cute.Pointer) -> cute.Swizzle: ...
@dsl_user_op
def fmax(
    a: float | Float32,
    b: float | Float32,
    c: float | Float32 | None = None,
    *,
    loc=None,
    ip=None,
) -> Float32: ...
@cute.jit
def fmax_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32: ...
@cute.jit
def fadd_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32: ...
@dsl_user_op
def atomic_add_fp32(
    a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None: ...
@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer: ...
@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor: ...
def canonical_warp_group_idx(sync: bool = True) -> cutlass.Int32: ...
@cute.jit
def shuffle_sync(
    value: cute.Numeric, offset: cute.typing.Int, width: cutlass.Constexpr[int] = ...
) -> cute.Numeric: ...
@dsl_user_op
def shr_u32(
    val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32: ...
@cute.jit
def warp_prefix_sum(
    val: cutlass.Int32, lane: cutlass.Int32 | None = None
) -> cutlass.Int32: ...
@dsl_user_op
def cvt_f16x2_f32(
    a: float | Float32, b: float | Float32, to_dtype: type, *, loc=None, ip=None
) -> cutlass.Int32: ...
@overload
def cvt_f16(src: cute.Tensor, dst: cute.Tensor) -> None: ...
@overload
def cvt_f16(src: cute.Tensor, dtype: type[cute.Numeric]) -> cute.Tensor: ...
@dsl_user_op
@cute.jit
def evaluate_polynomial(
    x: Float32, poly: tuple[Float32, ...], *, loc=None, ip=None
) -> Float32: ...
@dsl_user_op
@cute.jit
def evaluate_polynomial_2(
    x: Float32, y: Float32, poly: tuple[Float32, ...], *, loc=None, ip=None
) -> tuple[Float32, Float32]: ...
@dsl_user_op
def add_round_down(
    x: float | Float32, y: float | Float32, *, loc=None, ip=None
) -> Float32: ...
@dsl_user_op
def combine_int_frac_ex2(
    x_rounded: Float32, frac_ex2: Float32, *, loc=None, ip=None
) -> Float32: ...
@dsl_user_op
def ex2_emulation(x: Float32, *, loc=None, ip=None) -> Float32: ...
@dsl_user_op
def ex2_emulation_2(
    x: Float32, y: Float32, *, loc=None, ip=None
) -> tuple[Float32, Float32]: ...
@dsl_user_op
def e2e_asm2(
    x: Float32, y: Float32, *, loc=None, ip=None
) -> tuple[Float32, Float32]: ...
@dsl_user_op
def domain_offset_aligned(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor: ...
@cute.jit
def scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA: ...
def ssa_to_scalar(val): ...
