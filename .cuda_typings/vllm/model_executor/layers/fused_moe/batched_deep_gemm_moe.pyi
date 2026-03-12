import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from vllm.forward_context import (
    get_forward_context as get_forward_context,
    is_forward_context_available as is_forward_context_available,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate as TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8Dynamic128Sym as kFp8Dynamic128Sym,
    kFp8Static128BlockSym as kFp8Static128BlockSym,
)
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT as DeepGemmQuantScaleFMT,
    fp8_m_grouped_gemm_nt_masked as fp8_m_grouped_gemm_nt_masked,
    get_mk_alignment_for_contiguous_layout as get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_e8m0_used as is_deep_gemm_e8m0_used,
    is_deep_gemm_supported as is_deep_gemm_supported,
)
from vllm.utils.math_utils import cdiv as cdiv, round_up as round_up

logger: Incomplete

def scales_shape_stride_dtype(
    E: int, T: int, G: int, quant_scale_fmt: DeepGemmQuantScaleFMT
) -> tuple[tuple[int, ...], tuple[int, ...], torch.dtype]: ...
def persistent_masked_m_silu_mul_quant(
    y: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_parallel_tokens: int = 16,
    group_size: int = 128,
    quant_scale_fmt: DeepGemmQuantScaleFMT = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class BatchedDeepGemmExperts(mk.FusedMoEExpertsModular):
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
    def supports_packed_ue8m0_act_scales(self) -> bool: ...
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
    def estimate_expected_m(
        self, global_num_experts: int, max_tokens_per_expert: int, topk: int
    ) -> int: ...
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
