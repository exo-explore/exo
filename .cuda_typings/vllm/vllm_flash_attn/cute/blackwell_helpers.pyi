import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32
from vllm.vllm_flash_attn.cute.utils import (
    parse_swizzle_from_pointer as parse_swizzle_from_pointer,
)

@cute.jit
def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Int32 | None = None,
    B_idx: Int32 | None = None,
    zero_init: bool | Boolean = False,
    swap_AB: bool = False,
) -> None: ...
@cute.jit
def gemm_ptx_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: cute.Tensor | None,
    sB: cute.Tensor,
    A_idx: Int32 | None = None,
    B_idx: Int32 | None = None,
    zero_init: bool | Boolean = False,
    **kwargs,
) -> None: ...
@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: bool | Boolean = False,
) -> cute.TiledMma: ...
def i64_to_i32x2(i: int) -> tuple[int, int]: ...
@cute.jit
def gemm_ptx(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: cute.Tensor | None,
    sB: cute.Tensor,
    zero_init: bool | Boolean = False,
) -> None: ...
@cute.jit
def gemm_ptx_loop(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: cute.Tensor | None,
    sB: cute.Tensor,
    zero_init: bool | Boolean = False,
) -> None: ...
@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: cute.Tensor | None,
    sB: cute.Tensor,
    mbar_ptr: cutlass.Pointer | None = None,
    mbar_phase: Int32 | None = None,
    zero_init: bool | Boolean = False,
    tA_addr: Int32 | None = None,
) -> None: ...
@cute.jit
def gemm_ptx_partial1(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: cutlass.Constexpr[int],
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA_base_addr_for_desc: Int32,
    sA_addr_offset_for_desc: cutlass.Constexpr[int],
    sA_stage: Int32,
    sB_base_addr_for_desc: Int32,
    sB_addr_offset_for_desc: cutlass.Constexpr[int],
    sB_stage: Int32,
    sA_layout: cute.Layout | None,
    sB_layout: cute.Layout | None,
    sA_swizzle: cute.Swizzle | None,
    sB_swizzle: cute.Swizzle,
    zero_init: bool | Boolean = False,
) -> None: ...
