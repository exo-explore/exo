import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from _typeshed import Incomplete
from cutlass import Float32, Int32
from cutlass.pipeline import (
    PipelineAsync as PipelineAsync,
    PipelineConsumer as PipelineConsumer,
)
from typing import Callable
from vllm.vllm_flash_attn.cute import (
    barrier as barrier,
    copy_utils as copy_utils,
    pipeline as pipeline,
    utils as utils,
)
from vllm.vllm_flash_attn.cute.blackwell_helpers import (
    gemm_ptx_w_idx as gemm_ptx_w_idx,
    gemm_w_idx as gemm_w_idx,
)
from vllm.vllm_flash_attn.cute.block_info import BlockInfo as BlockInfo
from vllm.vllm_flash_attn.cute.block_sparse_utils import (
    get_block_sparse_iteration_info_bwd as get_block_sparse_iteration_info_bwd,
    get_m_block_from_iter_bwd as get_m_block_from_iter_bwd,
    get_total_q_block_count_bwd as get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm100 as produce_block_sparse_q_loads_bwd_sm100,
)
from vllm.vllm_flash_attn.cute.block_sparsity import (
    BlockSparseTensors as BlockSparseTensors,
)
from vllm.vllm_flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned as assume_tensor_aligned,
)
from vllm.vllm_flash_attn.cute.mask import AttentionMask as AttentionMask
from vllm.vllm_flash_attn.cute.named_barrier import (
    NamedBarrierBwdSm100 as NamedBarrierBwdSm100,
)
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK as SeqlenInfoQK
from vllm.vllm_flash_attn.cute.softmax import (
    apply_score_mod_bwd_inner as apply_score_mod_bwd_inner,
    apply_score_mod_inner as apply_score_mod_inner,
)
from vllm.vllm_flash_attn.cute.tile_scheduler import (
    ParamsBase as ParamsBase,
    SingleTileLPTBwdScheduler as SingleTileLPTBwdScheduler,
    SingleTileScheduler as SingleTileScheduler,
    SingleTileVarlenScheduler as SingleTileVarlenScheduler,
    TileSchedulerArguments as TileSchedulerArguments,
)

class FlashAttentionBackwardSm100:
    arch: int
    tile_hdim: Incomplete
    same_hdim_kv: Incomplete
    tile_hdimv: Incomplete
    check_hdim_oob: Incomplete
    check_hdim_v_oob: Incomplete
    tile_m: Incomplete
    tile_n: Incomplete
    cta_tiler: Incomplete
    mma_tiler_kq: Incomplete
    mma_tiler_vdo: Incomplete
    mma_tiler_pdo: Incomplete
    mma_tiler_dsq: Incomplete
    mma_tiler_dsk: Incomplete
    acc_dtype: Incomplete
    cluster_shape_mn: Incomplete
    is_persistent: Incomplete
    is_causal: Incomplete
    is_local: Incomplete
    qhead_per_kvhead: Incomplete
    pack_gqa: bool
    deterministic: Incomplete
    score_mod: Incomplete
    score_mod_bwd: Incomplete
    mask_mod: Incomplete
    has_aux_tensors: Incomplete
    subtile_factor: Incomplete
    vec_size: cutlass.Constexpr
    qk_acc_dtype: Incomplete
    shuffle_LSE: bool
    shuffle_dPsum: bool
    use_smem_dS_for_mma_dK: Incomplete
    reduce_warp_ids: Incomplete
    compute_warp_ids: Incomplete
    mma_warp_id: int
    load_warp_id: int
    epi_warp_id: int
    empty_warp_id: int
    threads_per_cta: Incomplete
    compute_sync_barrier: Incomplete
    reduce_sync_barrier: Incomplete
    tmem_alloc_cols: Incomplete
    tmem_S_offset: int
    tmem_P_offset: int
    tmem_dV_offset: Incomplete
    tmem_dP_offset: Incomplete
    tmem_dQ_offset: Incomplete
    tmem_dK_offset: Incomplete
    tmem_dS_offset: Incomplete
    num_regs_reduce: int
    num_regs_compute: int
    num_regs_other: Incomplete
    num_regs_empty: int
    buffer_align_bytes: int
    def __init__(
        self,
        head_dim: int,
        head_dim_v: int | None = None,
        is_causal: bool = False,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tile_m: int = 128,
        tile_n: int = 128,
        is_persistent: bool = False,
        deterministic: bool = False,
        cluster_size: int = 1,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        subtile_factor: cutlass.Constexpr[int] = 1,
    ) -> None: ...
    q_dtype: Incomplete
    k_dtype: Incomplete
    v_dtype: Incomplete
    do_dtype: Incomplete
    lse_dtype: Incomplete
    dpsum_dtype: Incomplete
    dqaccum_dtype: Incomplete
    dk_dtype: Incomplete
    dv_dtype: Incomplete
    ds_dtype: Incomplete
    is_varlen_k: Incomplete
    is_varlen_q: Incomplete
    use_tma_store: Incomplete
    dKV_postprocess: Incomplete
    cluster_shape_mnk: Incomplete
    cluster_layout_vmnk: Incomplete
    num_mcast_ctas_b: Incomplete
    is_q_do_mcast: Incomplete
    mdK_layout_enum: Incomplete
    mdV_layout_enum: Incomplete
    tma_copy_bytes: Incomplete
    spt: Incomplete
    tile_scheduler_cls: Incomplete
    shared_storage: Incomplete
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
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: cute.Tensor | None,
        mdK_tma_tensor: cute.Tensor | None,
        mdQ_semaphore: cute.Tensor | None,
        mdK_semaphore: cute.Tensor | None,
        mdV_semaphore: cute.Tensor | None,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        mSeqUsedK: cute.Tensor | None,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom | None,
        tma_atom_dK: cute.CopyAtom | None,
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        sdKV_layout: cute.ComposedLayout | cute.Layout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        window_size_left: Int32 | None,
        window_size_right: Int32 | None,
        tile_sched_params: ParamsBase,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: BlockSparseTensors | None = None,
    ): ...
    @cute.jit
    def load(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdO: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: BlockSparseTensors | None = None,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
    ): ...
    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        sdSt: cute.Tensor,
        sdS: cute.Tensor,
        sKt: cute.Tensor,
        tP: cute.Tensor,
        tdS: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        pipeline_Q_consumer: PipelineConsumer,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: BlockSparseTensors | None = None,
    ): ...
    @cute.jit
    def split_wg(
        self, t: cute.Tensor, wg_idx: cutlass.Int32, num_wg: cutlass.Constexpr[int]
    ): ...
    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_copy_t2r,
        thr_mma_S,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ) -> None: ...
    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor,
        score_tensor,
        index_tensor,
        batch_idx,
        head_idx,
        softmax_scale,
        seqlen_info,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ) -> None: ...
    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        tStS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdS: cute.Tensor,
        tdPtdP: cute.Tensor,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        sdV: cute.Tensor | None,
        sdK: cute.Tensor | None,
        mdV_tma_tensor: cute.Tensor | None,
        mdK_tma_tensor: cute.Tensor | None,
        tma_atom_dV: cute.CopyAtom | None,
        tma_atom_dK: cute.CopyAtom | None,
        tiled_copy_r2s_dKV: cute.TiledCopy | None,
        mdK_semaphore: cute.Tensor | None,
        mdV_semaphore: cute.Tensor | None,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: BlockSparseTensors | None = None,
    ): ...
    @cute.jit
    def dQacc_reduce(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        thr_mma_dQ: cute.core.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mdQ_semaphore: cute.Tensor | None,
        blocksparse_tensors: BlockSparseTensors | None = None,
    ): ...
    @cute.jit
    def epilogue_dKV(
        self,
        tidx: Int32,
        warp_idx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        softmax_scale: Float32,
    ): ...
    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        thr_mma: cute.core.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        scale: Float32 | None,
        barrier_id: Int32,
        mdKV_semaphore: cute.Tensor | None,
    ) -> cutlass.pipeline.PipelineState: ...
