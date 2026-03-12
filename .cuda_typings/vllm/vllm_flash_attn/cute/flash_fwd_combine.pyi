import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from _typeshed import Incomplete
from cutlass import Int32
from cutlass.cute import FastDivmodDivisor
from vllm.vllm_flash_attn.cute import utils as utils
from vllm.vllm_flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned as assume_tensor_aligned,
)
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfo as SeqlenInfo

class FlashAttentionForwardCombine:
    dtype: Incomplete
    dtype_partial: Incomplete
    head_dim: Incomplete
    m_block_size: Incomplete
    k_block_size: Incomplete
    max_splits: Incomplete
    num_threads: Incomplete
    is_even_k: Incomplete
    stages: Incomplete
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        dtype_partial: type[cutlass.Numeric],
        head_dim: int,
        m_block_size: int = 8,
        k_block_size: int = 64,
        log_max_splits: int = 4,
        num_threads: int = 256,
        stages: int = 4,
    ) -> None: ...
    @staticmethod
    def can_implement(
        dtype,
        dtype_partial,
        head_dim,
        m_block_size,
        k_block_size,
        log_max_splits,
        num_threads,
    ) -> bool: ...
    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None = None,
        cu_seqlens: cute.Tensor | None = None,
        seqused: cute.Tensor | None = None,
        num_splits_dynamic_ptr: cute.Tensor | None = None,
        semaphore_to_reset: cute.Tensor | None = None,
        stream: cuda.CUstream = None,
    ): ...
    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        cu_seqlens: cute.Tensor | None,
        seqused: cute.Tensor | None,
        num_splits_dynamic_ptr: cute.Tensor | None,
        semaphore_to_reset: cute.Tensor | None,
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        seqlen_divmod: FastDivmodDivisor,
        head_divmod: FastDivmodDivisor,
        varlen: cutlass.Constexpr[bool],
    ): ...
    @cute.jit
    def load_O_partial(
        self,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        tOrOptr: cute.Tensor,
        tOsO_partial: cute.Tensor,
        tOhidx: cute.Tensor,
        tOpO: cute.Tensor,
        tOcO: cute.Tensor,
        mO_cur_partial_layout: cute.Layout,
        split: Int32,
        stage: Int32,
    ) -> None: ...
