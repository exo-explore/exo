import torch
from _typeshed import Incomplete
from typing import Literal
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import ScalarType as ScalarType
from vllm.utils.flashinfer import (
    flashinfer_quant_nvfp4_8x4_sf_layout as flashinfer_quant_nvfp4_8x4_sf_layout,
)
from vllm.utils.math_utils import cdiv as cdiv

logger: Incomplete

def register_fake(fn): ...
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None: ...
def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None: ...
def paged_attention_rocm(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: torch.Tensor | None = None,
    mfma_type: str = ...,
) -> None: ...
def mla_decode_kvcache_cpu(
    out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None: ...
def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None: ...
def convert_vertical_slash_indexes(
    q_seqlens: torch.Tensor,
    kv_seqlens: torch.Tensor,
    vertical_indexes: torch.Tensor,
    slash_indexes: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
def convert_vertical_slash_indexes_mergehead(
    q_seqlens: torch.Tensor,
    kv_seqlens: torch.Tensor,
    vertical_indexes: torch.Tensor,
    slash_indexes: torch.Tensor,
    vertical_indices_count: torch.Tensor,
    slash_indices_count: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None: ...
def rms_norm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None: ...
def fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None: ...
def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
) -> None: ...
def apply_repetition_penalties_torch(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None: ...
def apply_repetition_penalties_cuda(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None: ...
def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None: ...
def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def rms_norm_per_block_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    group_size: list[int],
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
    tma_alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    split_k_iters: int,
    thx: int,
    thy: int,
) -> torch.Tensor: ...
def awq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
) -> torch.Tensor: ...
def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_exllama: bool,
    use_v2_format: bool,
    bit: int,
) -> torch.Tensor: ...
def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor, bit: int) -> None: ...
def cutlass_scaled_mm_supports_fp4(cuda_device_capability: int) -> bool: ...
def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor: ...
def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool: ...
def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool: ...
def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor: ...
def cutlass_scaled_mm_azp(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor: ...
def cutlass_sparse_scaled_mm_supported(cuda_device_capability: int) -> bool: ...
def cutlass_group_gemm_supported(cuda_device_capability: int) -> bool: ...
def cutlass_sparse_compress(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
def cutlass_scaled_sparse_mm(
    a: torch.Tensor,
    bt_nzs: torch.Tensor,
    bt_meta: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor: ...
def get_cutlass_moe_mm_data(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
    num_experts: int,
    n: int,
    k: int,
    blockscale_offsets: torch.Tensor | None = None,
): ...
def get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
    expert_first_token_offset: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    n: int,
    k: int,
    swap_ab: bool,
): ...
def shuffle_rows(input_tensor: torch.Tensor, dst2src_map: torch.Tensor): ...
def get_cutlass_batched_moe_mm_data(
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    num_local_experts: int,
    padded_m: int,
    n: int,
    k: int,
): ...
def cutlass_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    a_strides: torch.Tensor,
    b_strides: torch.Tensor,
    c_strides: torch.Tensor,
    per_act_token: bool,
    per_out_ch: bool,
): ...
def cutlass_fp4_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    alphas: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    sf_offsets: torch.Tensor,
): ...
def mxfp8_experts_quant(
    input_tensor: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
) -> None: ...
def cutlass_mxfp8_grouped_mm(
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_tensors: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
) -> None: ...
def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor: ...
def awq_marlin_repack(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor: ...
def gptq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor: ...
def awq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor: ...
def marlin_int4_fp8_preprocess(
    qweight: torch.Tensor,
    qzeros_or_none: torch.Tensor | None = None,
    inplace: bool = False,
): ...
def marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor: ...
def machete_supported_schedules(
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: torch.dtype | None,
    group_zeros_type: torch.dtype | None = None,
    channel_scales_type: torch.dtype | None = None,
    token_scales_type: torch.dtype | None = None,
    out_type: torch.dtype | None = None,
) -> list[str]: ...
def machete_mm(
    a: torch.Tensor,
    b_q: torch.Tensor,
    b_type: ScalarType,
    out_type: torch.dtype | None = None,
    b_group_scales: torch.Tensor | None = None,
    b_group_zeros: torch.Tensor | None = None,
    b_group_size: int | None = None,
    b_channel_scales: torch.Tensor | None = None,
    a_token_scales: torch.Tensor | None = None,
    schedule: str | None = None,
) -> torch.Tensor: ...
def machete_mm_fake(
    a: torch.Tensor,
    b_q: torch.Tensor,
    b_type: ScalarType,
    out_type: torch.dtype | None = None,
    b_group_scales: torch.Tensor | None = None,
    b_group_zeros: torch.Tensor | None = None,
    b_group_size: int | None = None,
    b_channel_scales: torch.Tensor | None = None,
    a_token_scales: torch.Tensor | None = None,
    schedule: str | None = None,
) -> torch.Tensor: ...
def machete_prepack_B(
    b_q_weight: torch.Tensor,
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: torch.dtype | None,
) -> torch.Tensor: ...
def machete_prepack_B_fake(
    b_q_weight: torch.Tensor,
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: torch.dtype | None,
) -> torch.Tensor: ...
def cutlass_w4a8_mm(
    a: torch.Tensor,
    b_q: torch.Tensor,
    b_group_scales: torch.Tensor,
    b_group_size: int,
    b_channel_scales: torch.Tensor,
    a_token_scales: torch.Tensor,
    out_type: torch.dtype | None = None,
    maybe_schedule: str | None = None,
) -> torch.Tensor: ...
def cutlass_w4a8_mm_fake(
    a: torch.Tensor,
    b_q: torch.Tensor,
    b_group_scales: torch.Tensor,
    b_group_size: int,
    b_channel_scales: torch.Tensor,
    a_token_scales: torch.Tensor,
    out_type: torch.dtype | None = None,
    maybe_schedule: str | None = None,
) -> torch.Tensor: ...
def cutlass_pack_scale_fp8(scales: torch.Tensor) -> torch.Tensor: ...
def cutlass_pack_scale_fp8_fake(scales: torch.Tensor) -> torch.Tensor: ...
def cutlass_encode_and_reorder_int4b(b: torch.Tensor) -> torch.Tensor: ...
def cutlass_encode_and_reorder_int4b_fake(b: torch.Tensor) -> torch.Tensor: ...
def cutlass_w4a8_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    b_group_scales: torch.Tensor,
    b_group_size: int,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    a_strides: torch.Tensor,
    b_strides: torch.Tensor,
    c_strides: torch.Tensor,
    group_scale_strides: torch.Tensor,
    maybe_schedule: str | None = None,
): ...
def cutlass_encode_and_reorder_int4b_grouped(
    b_tensors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def cutlass_encode_and_reorder_int4b_grouped_fake(b: torch.Tensor) -> torch.Tensor: ...
def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor: ...
def scaled_fp4_quant(
    input: torch.Tensor,
    input_global_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    backend: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]: ...
def scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def silu_and_mul_scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
    group_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def allspark_repack_weight(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    has_zp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def allspark_w8a16_gemm(
    a: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: torch.Tensor | None,
    n: int,
    group_size: int,
    sm_count: int,
    sm_version: int,
    CUBLAS_M_THRESHOLD: int,
    has_zp: bool,
    n32k16_reorder: bool,
) -> torch.Tensor: ...
def scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
def ggml_dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor: ...
def ggml_mul_mat_vec_a8(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor: ...
def ggml_mul_mat_a8(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor: ...
def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor: ...
def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: torch.SymInt,
    tokens: torch.SymInt,
) -> torch.Tensor: ...
def ggml_moe_get_block_size(quant_type: int) -> int: ...
def selective_scan_fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D_: torch.Tensor | None,
    z_: torch.Tensor | None,
    delta_bias_: torch.Tensor | None,
    delta_softplus: bool,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    ssm_states: torch.Tensor,
    pad_slot_id: int,
    block_size: int = 1024,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    cu_chunk_seqlen: torch.Tensor | None = None,
    last_chunk_indices: torch.Tensor | None = None,
): ...
def LLMM1(a: torch.Tensor, b: torch.Tensor, rows_per_block: int) -> torch.Tensor: ...
def wvSplitK(
    a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None
) -> torch.Tensor: ...
def wvSplitKrc(
    a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None
) -> torch.Tensor: ...
def wvSplitKQ(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    cu_count: int,
    bias: torch.Tensor = None,
) -> torch.Tensor: ...
def moe_sum(input: torch.Tensor, output: torch.Tensor): ...
def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None: ...
def batched_moe_align_block_size(
    max_tokens_per_batch: int,
    block_size: int,
    expert_num_tokens: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None: ...
def moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_loras: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    adapter_enabled: torch.Tensor,
    lora_ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None: ...
def moe_wna16_gemm(
    input: torch.Tensor,
    output: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    top_k: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    bit: int,
) -> torch.Tensor: ...
def router_gemm_bf16_fp32(
    input: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor: ...
def router_gemm_bf16_fp32_fake(
    input: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor: ...
def dsv3_router_gemm(
    hidden_states: torch.Tensor, router_weight: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor: ...
def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None: ...
def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None: ...
def grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
): ...
def moe_wna16_marlin_gemm(
    input: torch.Tensor,
    output: torch.Tensor | None,
    b_qweight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_qzeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_past_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
    thread_k: int = -1,
    thread_n: int = -1,
    blocks_per_sm: int = -1,
) -> torch.Tensor: ...
def marlin_gemm_moe_fake(
    a: torch.Tensor,
    b_q_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b_scales: torch.Tensor,
    b_zero_points: torch.Tensor,
    g_idx: torch.Tensor,
    perm: torch.Tensor,
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: torch.SymInt,
    size_n: torch.SymInt,
    size_k: torch.SymInt,
    is_k_full: bool,
    num_experts: int,
    topk: int,
    moe_block_size: int,
    replicate_input: bool,
    apply_weights: bool,
) -> torch.Tensor: ...
def moe_wna16_marlin_gemm_fake(
    input: torch.Tensor,
    output: torch.Tensor | None,
    b_qweight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_qzeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_past_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
): ...
def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None: ...
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None: ...
def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None: ...
def concat_and_cache_mla_rope_fused(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    kv_cache_scale: torch.Tensor,
) -> None: ...
def swap_blocks(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_size_in_bytes: int,
    block_mapping: torch.Tensor,
) -> None: ...
def convert_fp8(
    output: torch.Tensor, input: torch.Tensor, scale: float = 1.0, kv_dtype: str = "fp8"
) -> None: ...
def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None: ...
def cp_gather_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    seq_starts: torch.Tensor | None = None,
) -> None: ...
def cp_gather_and_upconvert_fp8_kv_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_starts: torch.Tensor,
    batch_size: int,
) -> None: ...
def concat_mla_q(
    ql_nope: torch.Tensor, q_pe: torch.Tensor, q_out: torch.Tensor
) -> None: ...
def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    kv_cache_dtype: str,
) -> None: ...
def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None: ...
def get_device_attribute(attribute: int, device: int) -> int: ...
def get_max_shared_memory_per_block_device_attribute(device: int) -> int: ...
def init_custom_ar(
    ipc_tensors: list[torch.Tensor],
    rank_data: torch.Tensor,
    rank: int,
    fully_connected: bool,
) -> int: ...
def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None: ...
def dispose(fa: int) -> None: ...
def meta_size() -> int: ...
def register_buffer(fa: int, ipc_tensors: list[int]) -> None: ...
def get_graph_buffer_ipc_meta(fa: int) -> tuple[list[int], list[int]]: ...
def register_graph_buffers(
    fa: int, handles: list[list[int]], offsets: list[list[int]]
) -> None: ...
def allocate_shared_buffer_and_handle(size: int) -> tuple[int, torch.Tensor]: ...
def open_mem_handle(mem_handle: torch.Tensor): ...
def free_shared_buffer(ptr: int) -> None: ...
def init_custom_qr(
    rank: int, world_size: int, qr_max_size: int | None = None
) -> int: ...
def qr_destroy(fa: int) -> None: ...
def qr_all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    quant_level: int,
    cast_bf2half: bool = False,
) -> None: ...
def qr_get_handle(fa: int) -> torch.Tensor: ...
def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None: ...
def qr_max_size() -> int: ...
def get_flash_mla_metadata(
    cache_seqlens: torch.Tensor, num_heads_per_head_k: int, num_heads_k: int
) -> tuple[torch.Tensor, torch.Tensor]: ...
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def sm100_cutlass_mla_decode(
    out: torch.Tensor,
    lse: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
    scale: float,
    num_kv_splits: int,
) -> torch.Tensor: ...
def sm100_cutlass_mla_get_workspace_size(
    max_seq_len: int, num_batches: int, sm_count: int, num_kv_splits: int
) -> int: ...
def dsv3_fused_a_gemm(
    output: torch.Tensor, mat_a: torch.Tensor, mat_b: torch.Tensor
) -> None: ...
def weight_packed_linear_fake(
    mat1: torch.Tensor, mat2: torch.Tensor, bias: torch.Tensor | None, is_vnni: bool
) -> torch.Tensor: ...
def fused_experts_cpu_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool,
    use_int8_w8a8: bool,
    use_fp8_w8a16: bool,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    block_size: list[int] | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    is_vnni: bool,
) -> torch.Tensor: ...
def int8_scaled_mm_with_quant_fake(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    scales2: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    is_vnni: bool,
) -> torch.Tensor: ...

class CPUDNNLGEMMHandler:
    handler_tensor: torch.Tensor | None
    n: int
    k: int
    def __init__(self) -> None: ...
    def __del__(self) -> None: ...

def is_onednn_acl_supported(): ...
def create_onednn_mm(
    weight: torch.Tensor, primitive_cache_size: int = 128
) -> CPUDNNLGEMMHandler: ...
def onednn_mm(
    dnnl_handler: CPUDNNLGEMMHandler, x: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor: ...
def create_onednn_scaled_mm(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    output_type: torch.dtype,
    dynamic_quant: bool,
    use_azp: bool,
    primitive_cache_size: int = 128,
) -> CPUDNNLGEMMHandler: ...
def onednn_scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
): ...
def onednn_scaled_mm(
    dnnl_handler: CPUDNNLGEMMHandler,
    x: torch.Tensor,
    output: torch.Tensor,
    input_scale: torch.Tensor | None,
    input_zp: torch.Tensor | None,
    input_zp_adj: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor: ...
def cpu_attn_get_scheduler_metadata(
    num_reqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: torch.Tensor,
    dtype: torch.dtype,
    query_start_loc: torch.Tensor,
    causal: bool,
    sliding_window_size: int,
    isa: str,
    enable_kv_split: bool,
) -> torch.Tensor: ...
def cpu_attn_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    isa: str,
) -> None: ...
def cpu_attention_with_kv_cache(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    causal: bool,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    block_table: torch.Tensor,
    softcap: float,
    scheduler_metadata: torch.Tensor,
    s_aux: torch.Tensor | None,
) -> None: ...
def cpu_gemm_wna16(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    bias: torch.Tensor | None,
    pack_factor: int,
    isa_hint: str,
) -> torch.Tensor: ...
def cpu_prepack_moe_weight(weight: torch.Tensor, isa: str) -> torch.Tensor: ...
def cpu_fused_moe(
    input: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    act: str,
    isa: str,
    skip_weighted: bool = False,
) -> torch.Tensor: ...
def matmul_mxf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor: ...
def matmul_ada_mxf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor: ...
def fusedQuantizeMx(
    a: torch.Tensor, b: torch.Tensor, *, method: Literal["quest", "abs_max"] = "quest"
) -> tuple[torch.Tensor, torch.Tensor]: ...
def fusedQuantizeNv(
    a: torch.Tensor, b: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
def hadacore_transform(x: torch.Tensor, inplace: bool = True) -> torch.Tensor: ...
