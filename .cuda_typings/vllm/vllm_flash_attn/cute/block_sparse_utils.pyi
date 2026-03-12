import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from typing import Callable
from vllm.vllm_flash_attn.cute import copy_utils as copy_utils
from vllm.vllm_flash_attn.cute.block_sparsity import (
    BlockSparseTensors as BlockSparseTensors,
)
from vllm.vllm_flash_attn.cute.named_barrier import NamedBarrierBwd as NamedBarrierBwd

@cute.jit
def load_block_list(
    block_indices: cute.Tensor,
    block_count,
    load_q_with_first: cutlass.Constexpr,
    first_block_preloaded: cutlass.Constexpr,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_k,
    pipeline_v,
    use_tma_q: cutlass.Constexpr,
    tma_q_bytes: cutlass.Constexpr,
    intra_wg_overlap: cutlass.Constexpr,
): ...
@cute.jit
def finish_overlap_v_load(
    block_indices: cute.Tensor, block_count, load_V, pipeline_v, kv_producer_state
): ...
@cute.jit
def sparse_tensor_m_block(
    m_block,
    qhead_per_kvhead: cutlass.Constexpr[int],
    q_subtile_factor: cutlass.Constexpr[int],
): ...
@cute.jit
def produce_block_sparse_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_k,
    pipeline_v,
    use_tma_q: cutlass.Constexpr,
    tma_q_bytes: cutlass.Constexpr,
    intra_wg_overlap: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
): ...
@cute.jit
def consume_block_sparse_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen,
    kv_consumer_state,
    mma_pv_fn,
    mma_one_n_block,
    process_first_half_block,
    process_last_half_block,
    mask_fn,
    score_mod_fn,
    O_should_accumulate,
    mask_mod,
    fastdiv_mods,
    intra_wg_overlap: cutlass.Constexpr,
    warp_scheduler_barrier_sync: Callable,
    warp_scheduler_barrier_arrive: Callable,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
): ...
@cute.jit
def load_block_list_sm100(
    block_indices: cute.Tensor,
    block_count,
    load_q_with_first: cutlass.Constexpr,
    m_block,
    q_stage: cutlass.Constexpr,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
): ...
@cute.jit
def produce_block_sparse_loads_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
    q_stage: cutlass.Constexpr,
    q_producer_phase: Int32,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
): ...
@cute.jit
def get_total_block_count(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
): ...
@cute.jit
def handle_block_sparse_empty_tile_correction_sm100(
    tidx: Int32,
    q_stage: cutlass.Constexpr,
    m_block_size: cutlass.Constexpr,
    qhead_per_kvhead,
    pack_gqa: cutlass.Constexpr,
    is_split_kv: cutlass.Constexpr,
    learnable_sink,
    mLSE,
    seqlen,
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    split_idx: Int32,
    sScale: cute.Tensor,
    stats: list,
    correction_epilogue: Callable,
    thr_mma_pv: cute.core.ThrMma,
    tOtOs: tuple[cute.Tensor],
    sO: cute.Tensor,
    mbar_ptr,
    mbar_softmax_corr_full_offset: Int32,
    mbar_softmax_corr_empty_offset: Int32,
    mbar_P_full_O_rescaled_offset: Int32,
    mbar_P_full_2_offset: Int32,
    mbar_corr_epi_full_offset: Int32,
    mbar_corr_epi_empty_offset: Int32,
    softmax_corr_consumer_phase: Int32,
    o_corr_consumer_phase: Int32,
    corr_epi_producer_phase: Int32,
    softmax_scale_log2: Float32,
    mO_cur: cute.Tensor | None = None,
    gO: cute.Tensor | None = None,
    gmem_tiled_copy_O: cute.TiledCopy | None = None,
): ...
@cute.jit
def softmax_block_sparse_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    softmax_step: Callable,
    mask_fn: Callable,
    mask_fn_none: Callable,
    mma_si_consumer_phase: Int32,
    si_corr_producer_phase: Int32,
    s0_s1_sequence_phase: Int32,
    mbar_ptr,
    mbar_softmax_corr_full_offset: Int32,
    mbar_softmax_corr_empty_offset: Int32,
    mbar_P_full_O_rescaled_offset: Int32,
    mbar_P_full_2_offset: Int32,
    q_stage: cutlass.Constexpr,
    stage_idx: Int32,
    check_m_boundary: bool,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
): ...
@cute.jit
def get_total_q_block_count_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
): ...
@cute.jit
def produce_block_sparse_q_loads_bwd_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    producer_state_Q_LSE,
    producer_state_dO_dPsum,
    pipeline_Q,
    pipeline_LSE,
    pipeline_dO,
    pipeline_dPsum,
    load_K,
    load_V,
    load_Q,
    load_dO,
    copy_stats,
    gLSE,
    sLSE,
    gdPsum,
    sdPsum,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    should_load_Q: cutlass.Constexpr,
    should_load_dO: cutlass.Constexpr,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
): ...
@cute.jit
def get_block_sparse_iteration_info_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
): ...
@cute.jit
def get_m_block_from_iter_bwd(
    iter_idx,
    curr_q_cnt,
    curr_q_idx: cute.Tensor,
    curr_full_cnt,
    curr_full_idx: cute.Tensor | None,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
): ...
@cute.jit
def produce_block_sparse_q_loads_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
): ...
@cute.jit
def consume_block_sparse_mma_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    consumer_state_Q,
    consumer_state_dO,
    mma_one_m_block_fn,
    mask,
    mask_mod,
    is_causal: cutlass.Constexpr,
    is_local: cutlass.Constexpr,
    thr_mma_SdP,
    score_mod_fn=None,
    score_mod_bwd_fn=None,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
    aux_tensors=None,
    fastdiv_mods=(None, None),
): ...
@cute.jit
def dQaccum_store_block_sparse_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
    num_mma_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
): ...
