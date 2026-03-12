import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from _typeshed import Incomplete
from cutlass import Float32 as Float32, Int32 as Int32
from types import SimpleNamespace
from typing import Callable
from vllm.vllm_flash_attn.cute import utils as utils
from vllm.vllm_flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned as assume_tensor_aligned,
)
from vllm.vllm_flash_attn.cute.mask import AttentionMask as AttentionMask
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK as SeqlenInfoQK
from vllm.vllm_flash_attn.cute.tile_scheduler import (
    ParamsBase as ParamsBase,
    SingleTileScheduler as SingleTileScheduler,
    SingleTileVarlenScheduler as SingleTileVarlenScheduler,
    TileSchedulerArguments as TileSchedulerArguments,
)

class FlashAttentionBackwardSm80:
    dtype: Incomplete
    head_dim_padded: Incomplete
    same_hdim_kv: Incomplete
    head_dim_v_padded: Incomplete
    check_hdim_oob: Incomplete
    check_hdim_v_oob: Incomplete
    qhead_per_kvhead: Incomplete
    m_block_size: Incomplete
    n_block_size: Incomplete
    num_threads: Incomplete
    pack_gqa: Incomplete
    is_causal: Incomplete
    num_stages_Q: Incomplete
    num_stages_dO: Incomplete
    SdP_swapAB: Incomplete
    dKV_swapAB: Incomplete
    dQ_swapAB: Incomplete
    AtomLayoutMSdP: Incomplete
    AtomLayoutNdKV: Incomplete
    AtomLayoutMdQ: Incomplete
    Mma_dKV_is_RS: Incomplete
    V_in_regs: Incomplete
    share_QV_smem: Incomplete
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int | None = None,
        qhead_per_kvhead: int = 1,
        m_block_size: int = 64,
        n_block_size: int = 128,
        num_stages_Q: int = 2,
        num_stages_dO: int = 2,
        num_threads: int = 256,
        pack_gqa: bool = False,
        is_causal: bool = False,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 8,
        AtomLayoutMdQ: int = 1,
        V_in_regs: bool = False,
    ) -> None: ...
    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
        n_block_size,
        num_stages_Q,
        num_stages_dO,
        num_threads,
        is_causal,
        V_in_regs: bool = False,
    ) -> bool: ...
    varlen_q: Incomplete
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mSeqUsedQ: cute.Tensor | None = None,
        mSeqUsedK: cute.Tensor | None = None,
        softcap: Float32 | float | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: cute.Tensor | None = None,
    ): ...
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        mSeqUsedK: cute.Tensor | None,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sLSEMma_layout: cute.Layout,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_VdO: cute.TiledCopy,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ): ...
    @cute.jit
    def compute_one_m_block(
        self,
        m_block: cutlass.Int32,
        smem_pipe_read_q: cutlass.Int32,
        smem_pipe_read_do: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        smem_pipe_write_do: cutlass.Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        load_Q_LSE: Callable,
        load_dO_dPsum: Callable,
        m_block_max: cutlass.Int32,
        softmax_scale_log2: cutlass.Float32,
        mask_fn: Callable | None = None,
    ): ...
    @cute.jit
    def epilogue(
        self,
        acc_dK: cute.Tensor,
        acc_dV: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sdK: cute.Tensor,
        sdV: cute.Tensor,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        n_block: cutlass.Int32,
        num_head: cutlass.Int32,
        batch_size: cutlass.Int32,
        seqlen: SeqlenInfoQK,
        d_head: cutlass.Int32,
        d_head_v: cutlass.Int32,
    ): ...
    @cute.jit
    def advance_pipeline(self, pipeline_index, num_stages: cutlass.Constexpr): ...
    @cute.jit
    def load_K(
        self,
        gmem_thr_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        headdim: cutlass.Int32,
    ): ...
    @cute.jit
    def load_V(
        self,
        gmem_thr_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        headdim: cutlass.Int32,
    ): ...
    @cute.jit
    def load_Q_LSE(
        self,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        tQgQ: cute.Tensor,
        tQsQ: cute.Tensor,
        tQcQ: cute.Tensor,
        t0QcQ: cute.Tensor,
        tQpQ: cute.Tensor,
        tLSEgLSE: cute.Tensor,
        tLSEsLSE: cute.Tensor,
        tLSEcLSE: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        seqlen: cutlass.Int32,
    ): ...
    @cute.jit
    def load_dO_dPsum(
        self,
        gmem_tiled_copy_dO: cute.TiledCopy,
        gmem_tiled_copy_dPsum: cute.TiledCopy,
        tdOgdO: cute.Tensor,
        tdOsdO: cute.Tensor,
        tdOcdO: cute.Tensor,
        t0dOcdO: cute.Tensor,
        tdOpdO: cute.Tensor,
        tdPsumgdPsum: cute.Tensor,
        tdPsumsdPsum: cute.Tensor,
        tdPsumcdPsum: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        seqlen: cutlass.Int32,
    ): ...
