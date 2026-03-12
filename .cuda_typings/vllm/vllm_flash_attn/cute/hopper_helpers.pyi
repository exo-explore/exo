import cutlass
import cutlass.cute as cute
from cutlass import Boolean as Boolean, Int32 as Int32
from cutlass.cutlass_dsl import Numeric as Numeric, dsl_user_op
from cutlass.utils import LayoutEnum as LayoutEnum

@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None: ...
def gemm_zero_init(
    tiled_mma: cute.TiledMma,
    shape: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Int32 | None = None,
    B_idx: Int32 | None = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> cute.Tensor: ...
def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: Boolean,
    A_idx: Int32 | None = None,
    B_idx: Int32 | None = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> None: ...
@dsl_user_op
def make_smem_layout(
    dtype: type[Numeric],
    layout: LayoutEnum,
    shape: cute.Shape,
    stage: int | None = None,
    *,
    loc=None,
    ip=None,
) -> cute.Layout | cute.ComposedLayout: ...
