import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from _typeshed import Incomplete
from cutlass import Boolean as Boolean, Constexpr as Constexpr, Float32, Int32
from types import SimpleNamespace
from typing import Callable
from vllm.vllm_flash_attn.cute import pipeline as pipeline, utils as utils
from vllm.vllm_flash_attn.cute.block_info import BlockInfo as BlockInfo
from vllm.vllm_flash_attn.cute.block_sparse_utils import (
    consume_block_sparse_loads as consume_block_sparse_loads,
    produce_block_sparse_loads as produce_block_sparse_loads,
)
from vllm.vllm_flash_attn.cute.block_sparsity import (
    BlockSparseTensors as BlockSparseTensors,
)
from vllm.vllm_flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned as assume_tensor_aligned,
)
from vllm.vllm_flash_attn.cute.mask import AttentionMask as AttentionMask
from vllm.vllm_flash_attn.cute.named_barrier import NamedBarrierFwd as NamedBarrierFwd
from vllm.vllm_flash_attn.cute.pack_gqa import PackGQA as PackGQA
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK as SeqlenInfoQK
from vllm.vllm_flash_attn.cute.softmax import (
    Softmax as Softmax,
    apply_score_mod_inner as apply_score_mod_inner,
)
from vllm.vllm_flash_attn.cute.tile_scheduler import (
    ParamsBase as ParamsBase,
    SingleTileLPTScheduler as SingleTileLPTScheduler,
    SingleTileScheduler as SingleTileScheduler,
    SingleTileVarlenScheduler as SingleTileVarlenScheduler,
    TileSchedulerArguments as TileSchedulerArguments,
)

class FlashAttentionForwardBase:
    arch: int
    dtype: Incomplete
    tile_hdim: Incomplete
    same_hdim_kv: Incomplete
    tile_hdimv: Incomplete
    check_hdim_oob: Incomplete
    check_hdim_v_oob: Incomplete
    qhead_per_kvhead: Incomplete
    is_causal: Incomplete
    is_local: Incomplete
    pack_gqa: Incomplete
    tile_m: Incomplete
    tile_n: Incomplete
    num_threads: Incomplete
    num_stages: Incomplete
    q_subtile_factor: Incomplete
    Q_in_regs: Incomplete
    score_mod: Incomplete
    mask_mod: Incomplete
    qk_acc_dtype: Incomplete
    vec_size: cutlass.Constexpr
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int | None = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = True,
        tile_m: int = 128,
        tile_n: int = 128,
        num_stages: int = 1,
        num_threads: int = 128,
        Q_in_regs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: bool = False,
        q_subtile_factor: int | None = None,
    ) -> None: ...
    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs: bool = False,
    ) -> bool: ...
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
    ): ...
    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom | None,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ): ...
    @cute.jit
    def advance_pipeline(self, pipeline_index): ...
    @cute.jit
    def load_Q(
        self,
        gmem_thr_copy: cute.TiledCopy,
        gQ: cute.Tensor,
        sQ: cute.Tensor,
        block: Int32,
        seqlen: Int32,
        headdim: Int32,
    ): ...
    @cute.jit
    def load_K(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tKcK: cute.Tensor,
        t0KcK: cute.Tensor,
        tKpK: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ): ...
    @cute.jit
    def load_V(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        tVcV: cute.Tensor,
        t0VcV: cute.Tensor,
        tVpV: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ): ...

class FlashAttentionForwardSm80(FlashAttentionForwardBase):
    num_mma_threads: Incomplete
    num_producer_threads: Incomplete
    num_Q_load_threads: Incomplete
    num_epilogue_threads: Incomplete
    use_tma_O: Incomplete
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        stream: cuda.CUstream,
        softmax_scale: Float32 | None = None,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
        learnable_sink: cute.Tensor | None = None,
        aux_tensors=None,
    ): ...
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Int32 | None,
        window_size_right: Int32 | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        aux_tensors=None,
        fastdiv_mods=None,
    ): ...
    @cute.jit
    def compute_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: Int32,
        smem_pipe_write: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        score_mod: Callable | None,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=None,
        mask_fn: Callable | None = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ): ...

class FlashAttentionForwardSm90(FlashAttentionForwardBase):
    arch: int
    intra_wg_overlap: Incomplete
    mma_pv_is_rs: Incomplete
    buffer_align_bytes: int
    def __init__(
        self, *args, intra_wg_overlap: bool = True, mma_pv_is_rs: bool = True, **kwargs
    ) -> None: ...
    num_mma_threads: Incomplete
    num_threads_per_warp_group: int
    num_mma_warp_groups: Incomplete
    num_threads: Incomplete
    num_producer_threads: int
    num_Q_load_threads: Incomplete
    num_epilogue_threads: Incomplete
    num_mma_regs: Incomplete
    num_producer_regs: Incomplete
    use_block_sparsity: Incomplete
    use_scheduler_barrier: Incomplete
    use_tma_Q: Incomplete
    use_tma_O: Incomplete
    sP_layout: Incomplete
    tma_copy_bytes: Incomplete
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
        tma_atom_Q: cute.CopyAtom | None,
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
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_tensors=...,
        fastdiv_mods=None,
    ): ...
    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        blocksparse_tensors: BlockSparseTensors | None,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ): ...
    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: cute.Tensor | None,
        sO: cute.Tensor,
        learnable_sink: cute.Tensor | None,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom | None,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: BlockSparseTensors | None,
        aux_tensors: list | None,
        fastdiv_mods=None,
    ): ...
    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        mask_fn: Callable = None,
        score_mod_fn: Callable | None = None,
        is_first_block: bool = False,
    ): ...
    @cute.jit
    def last_half_block_overlap(
        self, kv_consumer_state, pipeline_v, mma_pv_fn: Callable, zero_init: bool
    ): ...
    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        score_mod_fn: Callable | None = None,
        mask_fn: Callable | None = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ): ...
    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        score_mod_fn: Callable | None = None,
        mask_fn: Callable | None = None,
        check_inf: cutlass.Constexpr = True,
    ): ...
    @cute.jit
    def mma_init(self) -> None: ...
    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_tensors: list | None = None,
        fastdiv_mods=None,
    ): ...
    def warp_scheduler_barrier_sync(self) -> None: ...
    def warp_scheduler_barrier_arrive(self) -> None: ...
