import cutlass
import cutlass.cute as cute
from typing import Callable

def get_smem_layout_atom(
    dtype: type[cutlass.Numeric], k_dim: int
) -> cute.ComposedLayout: ...
@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Callable | None = None,
    A_in_regs: cutlass.Constexpr[bool] = False,
    B_in_regs: cutlass.Constexpr[bool] = False,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None: ...
@cute.jit
def gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Callable | None = None,
) -> None: ...
