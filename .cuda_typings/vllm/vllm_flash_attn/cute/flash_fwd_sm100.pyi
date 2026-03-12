import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import enum
from _typeshed import Incomplete
from cutlass import Float32, Int32
from typing import Callable, Literal
from vllm.vllm_flash_attn.cute import copy_utils as copy_utils
from vllm.vllm_flash_attn.cute.block_info import BlockInfo as BlockInfo
from vllm.vllm_flash_attn.cute.block_sparse_utils import (
    get_total_block_count as get_total_block_count,
    handle_block_sparse_empty_tile_correction_sm100 as handle_block_sparse_empty_tile_correction_sm100,
    produce_block_sparse_loads_sm100 as produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100 as softmax_block_sparse_sm100,
)
from vllm.vllm_flash_attn.cute.block_sparsity import (
    BlockSparseTensors as BlockSparseTensors,
)
from vllm.vllm_flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned as assume_tensor_aligned,
)
from vllm.vllm_flash_attn.cute.mask import AttentionMask as AttentionMask
from vllm.vllm_flash_attn.cute.pack_gqa import PackGQA as PackGQA
from vllm.vllm_flash_attn.cute.paged_kv import PagedKVManager as PagedKVManager
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK as SeqlenInfoQK
from vllm.vllm_flash_attn.cute.softmax import (
    SoftmaxSm100 as SoftmaxSm100,
    apply_score_mod_inner as apply_score_mod_inner,
)
from vllm.vllm_flash_attn.cute.tile_scheduler import (
    ParamsBase as ParamsBase,
    SingleTileLPTScheduler as SingleTileLPTScheduler,
    SingleTileScheduler as SingleTileScheduler,
    SingleTileVarlenScheduler as SingleTileVarlenScheduler,
    StaticPersistentTileScheduler as StaticPersistentTileScheduler,
    TileSchedulerArguments as TileSchedulerArguments,
)

class NamedBarrierFwd(enum.IntEnum):
    Epilogue = ...

class FlashAttentionForwardSm100:
    arch: int
    use_tma_KV: Incomplete
    head_dim_padded: Incomplete
    same_hdim_kv: Incomplete
    head_dim_v_padded: Incomplete
    same_hdim_kv_padded: Incomplete
    check_hdim_oob: Incomplete
    check_hdim_v_oob: Incomplete
    m_block_size: Incomplete
    n_block_size: Incomplete
    q_stage: Incomplete
    cta_tiler: Incomplete
    mma_tiler_qk: Incomplete
    mma_tiler_pv: Incomplete
    qk_acc_dtype: Incomplete
    pv_acc_dtype: Incomplete
    cluster_shape_mn: Incomplete
    is_persistent: Incomplete
    is_causal: Incomplete
    is_local: Incomplete
    is_varlen_q: Incomplete
    use_correction_warps_for_epi: Incomplete
    qhead_per_kvhead: Incomplete
    is_split_kv: Incomplete
    pack_gqa: Incomplete
    q_subtile_factor: Incomplete
    score_mod: Incomplete
    mask_mod: Incomplete
    vec_size: cutlass.Constexpr
    s0_s1_barrier: bool
    overlap_sO_sQ: Incomplete
    softmax0_warp_ids: Incomplete
    softmax1_warp_ids: Incomplete
    correction_warp_ids: Incomplete
    mma_warp_id: int
    epilogue_warp_ids: Incomplete
    load_warp_ids: Incomplete
    empty_warp_ids: Incomplete
    tmem_alloc_cols: Incomplete
    threads_per_cta: Incomplete
    tmem_s_offset: Incomplete
    tmem_o_offset: Incomplete
    tmem_total: Incomplete
    tmem_s_to_p_offset: Incomplete
    tmem_p_offset: Incomplete
    tmem_vec_offset: Incomplete
    num_regs_softmax: Incomplete
    num_regs_correction: int
    num_regs_other: Incomplete
    num_regs_empty: int
    buffer_align_bytes: int
    def __init__(
        self,
        head_dim: int,
        head_dim_v: int | None = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int | None = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: cutlass.Constexpr[int] = 2,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
    ) -> None: ...
    q_dtype: Incomplete
    k_dtype: Incomplete
    v_dtype: Incomplete
    o_dtype: Incomplete
    q_major_mode: Incomplete
    k_major_mode: Incomplete
    v_major_mode: Incomplete
    o_layout: Incomplete
    use_tma_O: Incomplete
    e2e_freq: int
    cluster_shape_mnk: Incomplete
    cluster_layout_vmnk: Incomplete
    epi_tile: Incomplete
    tma_copy_bytes: Incomplete
    num_epilogue_threads: Incomplete
    tile_scheduler_cls: Incomplete
    mbar_load_q_full_offset: int
    mbar_load_q_empty_offset: Incomplete
    mbar_load_kv_full_offset: Incomplete
    mbar_load_kv_empty_offset: Incomplete
    mbar_P_full_O_rescaled_offset: Incomplete
    mbar_S_full_offset: Incomplete
    mbar_O_full_offset: Incomplete
    mbar_softmax_corr_full_offset: Incomplete
    mbar_softmax_corr_empty_offset: Incomplete
    mbar_corr_epi_full_offset: Incomplete
    mbar_corr_epi_empty_offset: Incomplete
    mbar_s0_s1_sequence_offset: Incomplete
    mbar_tmem_dealloc_offset: Incomplete
    mbar_P_full_2_offset: Incomplete
    mbar_total: Incomplete
    shared_storage: Incomplete
    use_block_sparsity: Incomplete
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mSeqUsedQ: cute.Tensor | None = None,
        mSeqUsedK: cute.Tensor | None = None,
        mPageTable: cute.Tensor | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: cute.Tensor | None = None,
        blocksparse_tensors: BlockSparseTensors | None = None,
        aux_tensors: list | None = None,
    ): ...
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        mSeqUsedK: cute.Tensor | None,
        mPageTable: cute.Tensor | None,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom | None,
        tma_atom_V: cute.CopyAtom | None,
        tma_atom_O: cute.CopyAtom | None,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Int32 | None,
        window_size_right: Int32 | None,
        learnable_sink: cute.Tensor | None,
        blocksparse_tensors: BlockSparseTensors | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: cute.TiledCopy | None,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
    ): ...
    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: cute.Tensor | None,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom | None,
        tma_atom_V: cute.CopyAtom | None,
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: BlockSparseTensors | None,
    ): ...
    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStSs: tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: BlockSparseTensors | None,
    ): ...
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: cute.Tensor | None,
        learnable_sink: cute.Tensor | None,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        blocksparse_tensors: BlockSparseTensors | None = None,
    ): ...
    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.core.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Callable | None = None,
        is_first: bool = False,
    ) -> tuple[cute.Int32, cute.Int32, cute.Int32]: ...
    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOtOs: tuple[cute.Tensor],
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        learnable_sink: cute.Tensor | None,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        mbar_ptr: cute.Pointer,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: BlockSparseTensors | None = None,
    ): ...
    @cute.jit
    def correction_rescale(
        self, thr_mma: cute.core.ThrMma, tOtO: cute.Tensor, tidx: Int32, scale: Float32
    ): ...
    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: cute.Tensor | None = None,
        gO: cute.Tensor | None = None,
        gmem_tiled_copy_O: cute.TiledCopy | None = None,
    ): ...
    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom | None,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ): ...
    def load_Q(
        self,
        load_Q_fn: Callable,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        stage: int,
        phase: Int32,
    ): ...
    @cute.jit
    def load_KV(
        self,
        tma_atom: cute.CopyAtom | None,
        tXgX: cute.Tensor | None,
        tXsX: cute.Tensor | None,
        paged_kv_manager: PagedKVManager | None,
        sX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Int32 | None = None,
    ): ...
    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32): ...
    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr): ...
    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
        head_divmod=None,
    ): ...
