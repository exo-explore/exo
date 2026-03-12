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

class FlashAttentionBackwardPreprocess:
    dtype: Incomplete
    m_block_size: Incomplete
    arch: Incomplete
    head_dim_padded: Incomplete
    check_hdim_oob: Incomplete
    num_threads: Incomplete
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        arch: Literal[80, 90, 100],
        m_block_size: int = 128,
        num_threads: int = 128,
    ) -> None: ...
    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, num_threads) -> bool: ...
    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSE: cute.Tensor | None,
        mLSElog2: cute.Tensor | None,
        mdQaccum: cute.Tensor | None,
        mCuSeqlensQ: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        stream: cuda.CUstream,
    ): ...
    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSE: cute.Tensor | None,
        mLSElog2: cute.Tensor | None,
        mdQaccum: cute.Tensor | None,
        mCuSeqlensQ: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ): ...
