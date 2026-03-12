import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from vllm import envs as envs
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
    kNvfp4Dynamic as kNvfp4Dynamic,
    kNvfp4Static as kNvfp4Static,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.flashinfer import (
    flashinfer_cutedsl_grouped_gemm_nt_masked as flashinfer_cutedsl_grouped_gemm_nt_masked,
    scaled_fp4_grouped_quantize as scaled_fp4_grouped_quantize,
    silu_and_mul_scaled_nvfp4_experts_quantize as silu_and_mul_scaled_nvfp4_experts_quantize,
)

logger: Incomplete

class FlashInferCuteDSLExperts(mk.FusedMoEExpertsModular):
    out_dtype: Incomplete
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ) -> None: ...
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_expert_map(self) -> bool: ...
    def supports_chunking(self) -> bool: ...
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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ): ...

def get_cute_dtype(input: torch.Tensor) -> str: ...
def flashinfer_cutedsl_moe_masked(
    hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    input_global_scale: torch.Tensor,
    w1: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alpha,
    w2: torch.Tensor,
    a2_global_scale: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alpha,
    masked_m: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
): ...
