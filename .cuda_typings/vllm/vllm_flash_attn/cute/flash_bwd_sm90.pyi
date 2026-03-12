import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from _typeshed import Incomplete
from cutlass import Boolean as Boolean, Float32, Int32
from cutlass.cute import FastDivmodDivisor
from typing import Callable
from vllm.vllm_flash_attn.cute import pipeline as pipeline, utils as utils
from vllm.vllm_flash_attn.cute.block_info import BlockInfo as BlockInfo
from vllm.vllm_flash_attn.cute.block_sparse_utils import (
    consume_block_sparse_mma_bwd_sm90 as consume_block_sparse_mma_bwd_sm90,
    dQaccum_store_block_sparse_bwd_sm90 as dQaccum_store_block_sparse_bwd_sm90,
    get_total_q_block_count_bwd as get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm90 as produce_block_sparse_q_loads_bwd_sm90,
)
from vllm.vllm_flash_attn.cute.block_sparsity import (
    BlockSparseTensors as BlockSparseTensors,
)
from vllm.vllm_flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned as assume_tensor_aligned,
)
from vllm.vllm_flash_attn.cute.mask import AttentionMask as AttentionMask
from vllm.vllm_flash_attn.cute.named_barrier import (
    NamedBarrierBwd as NamedBarrierBwd,
    NamedBarrierFwd as NamedBarrierFwd,
)
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK as SeqlenInfoQK
from vllm.vllm_flash_attn.cute.softmax import (
    apply_score_mod_bwd_inner as apply_score_mod_bwd_inner,
    apply_score_mod_inner as apply_score_mod_inner,
)
from vllm.vllm_flash_attn.cute.tile_scheduler import (
    ParamsBase as ParamsBase,
    SingleTileScheduler as SingleTileScheduler,
    TileSchedulerArguments as TileSchedulerArguments,
)

class FlashAttentionBackwardSm90:
    arch: int
    dtype: Incomplete
    tile_hdim: Incomplete
    same_hdim_kv: Incomplete
    tile_hdimv: Incomplete
    check_hdim_oob: Incomplete
    check_hdim_v_oob: Incomplete
    qhead_per_kvhead: Incomplete
    is_causal: Incomplete
    is_local: bool
    tile_m: Incomplete
    tile_n: Incomplete
    num_threads: Incomplete
    Q_stage: Incomplete
    dO_stage: Incomplete
    PdS_stage: Incomplete
    SdP_swapAB: Incomplete
    dKV_swapAB: Incomplete
    dQ_swapAB: Incomplete
    AtomLayoutMSdP: Incomplete
    AtomLayoutNdKV: Incomplete
    AtomLayoutMdQ: Incomplete
    num_mma_warp_groups: Incomplete
    mma_dkv_is_rs: Incomplete
    V_in_regs: Incomplete
    shuffle_LSE: Incomplete
    shuffle_dPsum: Incomplete
    score_mod: Incomplete
    score_mod_bwd: Incomplete
    mask_mod: Incomplete
    has_aux_tensors: Incomplete
    subtile_factor: Incomplete
    vec_size: cutlass.Constexpr
    qk_acc_dtype: Incomplete
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int | None = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        tile_m: int = 64,
        tile_n: int = 128,
        Q_stage: int = 2,
        dO_stage: int = 2,
        PdS_stage: int = 2,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 2,
        AtomLayoutMdQ: int = 1,
        num_threads: int = 384,
        V_in_regs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        subtile_factor: cutlass.Constexpr[int] = 1,
    ) -> None: ...
    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        Q_stage,
        num_threads,
        V_in_regs: bool = False,
    ) -> bool: ...
    num_mma_threads: Incomplete
    num_threads_per_warp_group: int
    num_producer_threads: int
    num_mma_regs: int
    num_producer_regs: int
    tma_copy_bytes: Incomplete
    use_block_sparsity: Incomplete
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
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mSeqUsedQ: cute.Tensor | None = None,
        mSeqUsedK: cute.Tensor | None = None,
        softcap: Float32 | float | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: cute.Tensor | None = None,
        mdK_semaphore: cute.Tensor | None = None,
        mdV_semaphore: cute.Tensor | None = None,
        aux_tensors: list | None = None,
        blocksparse_tensors: BlockSparseTensors | None = None,
    ): ...
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        sdKVaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        r2s_tiled_copy_dKVaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: BlockSparseTensors | None = None,
        qhead_per_kvhead_divmod: FastDivmodDivisor | None = None,
    ): ...
    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: BlockSparseTensors | None = None,
        qhead_per_kvhead_divmod: FastDivmodDivisor | None = None,
    ): ...
    @cute.jit
    def apply_score_mod(
        self,
        acc_S: cute.Tensor,
        thr_mma_SdP: cute.core.ThrMma,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ): ...
    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor: cute.Tensor,
        score_tensor: cute.Tensor,
        thr_mma_SdP: cute.core.ThrMma,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ): ...
    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: cute.Tensor | None,
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        r2s_tiled_copy_dKVaccum: cute.TiledCopy,
        sdKVaccum_layout: cute.Layout,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: BlockSparseTensors | None = None,
        qhead_per_kvhead_divmod: FastDivmodDivisor | None = None,
    ): ...
    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state_Q: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_dO: cutlass.pipeline.PipelineState
        | pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tPsP: cute.Tensor | None,
        tdSsdS: cute.Tensor | None,
        tdQsdQaccum: cute.Tensor,
        smem_thr_copy_PdS: cute.TiledCopy,
        smem_thr_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        mask_fn: Callable | None = None,
        score_mod_fn: Callable | None = None,
        score_mod_bwd_fn: Callable | None = None,
        dKV_accumulate: Boolean = True,
    ): ...
    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        r2s_tiled_copy_dKVaccum: cute.TiledCopy,
        sdKVaccum_layout: cute.Layout,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        qhead_per_kvhead_divmod: FastDivmodDivisor | None = None,
    ): ...
    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        block_info: BlockInfo,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
        blocksparse_tensors: BlockSparseTensors | None = None,
    ): ...
