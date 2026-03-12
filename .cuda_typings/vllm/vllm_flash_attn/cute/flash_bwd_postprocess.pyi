import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from _typeshed import Incomplete
from typing import Callable, Literal
from vllm.vllm_flash_attn.cute import copy_utils as copy_utils, utils as utils
from vllm.vllm_flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned as assume_tensor_aligned,
)
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK as SeqlenInfoQK
from vllm.vllm_flash_attn.cute.tile_scheduler import (
    ParamsBase as ParamsBase,
    SingleTileScheduler as SingleTileScheduler,
    SingleTileVarlenScheduler as SingleTileVarlenScheduler,
    TileSchedulerArguments as TileSchedulerArguments,
)

class FlashAttentionBackwardPostprocess:
    dtype: Incomplete
    tile_m: Incomplete
    arch: Incomplete
    tile_hdim: Incomplete
    check_hdim_oob: Incomplete
    num_threads: Incomplete
    AtomLayoutMdQ: Incomplete
    dQ_swapAB: Incomplete
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        arch: Literal[80, 90, 100],
        tile_m: int = 128,
        num_threads: int = 256,
        AtomLayoutMdQ: int = 1,
        dQ_swapAB: bool = False,
    ) -> None: ...
    @staticmethod
    def can_implement(dtype, head_dim, tile_m, num_threads) -> bool: ...
    tiled_mma: Incomplete
    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        mCuSeqlensQ: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        stream: cuda.CUstream,
    ): ...
    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        mCuSeqlensQ: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        dQ_swapAB: cutlass.Constexpr,
        sdQaccum_layout: cute.Layout,
        sdQ_layout: cute.ComposedLayout,
        g2s_tiled_copy_dQaccum: cute.TiledCopy,
        s2r_tiled_copy_dQaccum: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ): ...
