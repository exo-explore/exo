import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M as compute_aligned_M,
    deepgemm_moe_permute as deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce as deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP as TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8 as per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm as per_token_group_quant_fp8_packed_for_deepgemm,
    silu_mul_per_token_group_quant_fp8_colmajor as silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8Dynamic128Sym as kFp8Dynamic128Sym,
    kFp8Static128BlockSym as kFp8Static128BlockSym,
)
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT as DeepGemmQuantScaleFMT,
    get_mk_alignment_for_contiguous_layout as get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_supported as is_deep_gemm_supported,
    m_grouped_fp8_gemm_nt_contiguous as m_grouped_fp8_gemm_nt_contiguous,
)
from vllm.utils.import_utils import has_deep_gemm as has_deep_gemm

logger: Incomplete

class DeepGemmExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig
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
