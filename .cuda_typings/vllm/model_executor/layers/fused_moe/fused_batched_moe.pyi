import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    try_get_optimal_moe_config as try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate as TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNaiveBatched as TopKWeightAndReduceNaiveBatched,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input as moe_kernel_quantize_input,
    normalize_batched_scales_shape as normalize_batched_scales_shape,
    normalize_scales_shape as normalize_scales_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    group_broadcast as group_broadcast,
    kFp8Dynamic128Sym as kFp8Dynamic128Sym,
    kFp8DynamicTensorSym as kFp8DynamicTensorSym,
    kFp8DynamicTokenSym as kFp8DynamicTokenSym,
    kFp8Static128BlockSym as kFp8Static128BlockSym,
    kFp8StaticChannelSym as kFp8StaticChannelSym,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
)
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import tl as tl, triton as triton

@triton.jit
def moe_mmk(
    a_ptrs,
    b_ptrs,
    K,
    expert_id,
    a_scale_ptr,
    b_scale_ptr,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    offs_m,
    offs_n,
    offs_bn,
    mask_m,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
    use_w8a8: tl.constexpr,
    use_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
): ...
@triton.jit
def expert_triton_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    expert_id,
    compute_type: tl.constexpr,
    M,
    N,
    K,
    a_scale_ptr,
    b_scale_ptr,
    b_zp_ptr,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    offs_bn,
    group_n,
    group_k,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
): ...
@triton.jit
def batched_triton_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    expert_num_tokens,
    compute_type: tl.constexpr,
    max_num_tokens,
    K,
    N,
    a_scale_ptr,
    b_scale_ptr,
    b_zp_ptr,
    stride_ae: tl.int64,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_be: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_ce: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
): ...
def invoke_moe_batched_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    compute_type: tl.dtype,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    config: dict[str, int],
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
): ...

class BatchedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    max_num_tokens: Incomplete
    num_local_experts: Incomplete
    rank: Incomplete
    num_dispatchers_: Incomplete
    def __init__(
        self,
        max_num_tokens: int,
        num_local_experts: int,
        num_dispatchers: int,
        rank: int,
    ) -> None: ...
    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat: ...
    def max_num_tokens_per_rank(self) -> int | None: ...
    def topk_indices_dtype(self) -> torch.dtype | None: ...
    def num_dispatchers(self) -> int: ...
    def output_is_reduced(self) -> bool: ...
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType: ...
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None: ...

class NaiveBatchedExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ) -> None: ...
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_chunking(self) -> bool: ...
    def supports_expert_map(self) -> bool: ...
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce: ...
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
    def dequant(self, t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor: ...
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ): ...

def batched_moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    num_tokens: int,
    E: int,
    N: int,
    expert_num_tokens: torch.Tensor,
    qtype: torch.dtype | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class BatchedTritonExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ) -> None: ...
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_chunking(self) -> bool: ...
    def supports_expert_map(self) -> bool: ...
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce: ...
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ): ...
