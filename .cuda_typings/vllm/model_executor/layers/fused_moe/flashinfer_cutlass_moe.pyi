import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from vllm.config import get_current_vllm_config as get_current_vllm_config
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP as TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8Dynamic128Sym as kFp8Dynamic128Sym,
    kFp8Static128BlockSym as kFp8Static128BlockSym,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
    kMxfp4Static as kMxfp4Static,
    kMxfp8Dynamic as kMxfp8Dynamic,
    kNvfp4Dynamic as kNvfp4Dynamic,
    kNvfp4Static as kNvfp4Static,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.flashinfer import (
    flashinfer_cutlass_fused_moe as flashinfer_cutlass_fused_moe,
    has_flashinfer_cutlass_fused_moe as has_flashinfer_cutlass_fused_moe,
)

logger: Incomplete

def is_valid_flashinfer_cutlass_fused_moe(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool: ...

class FlashInferExperts(mk.FusedMoEExpertsModular):
    device: Incomplete
    num_experts: Incomplete
    ep_rank: Incomplete
    ep_size: Incomplete
    tp_rank: Incomplete
    tp_size: Incomplete
    out_dtype: Incomplete
    use_dp: Incomplete
    use_deepseek_fp8_block_scale: Incomplete
    max_capture_size: Incomplete
    gemm1_alpha: Incomplete
    gemm1_beta: Incomplete
    gemm1_clamp_limit: Incomplete
    fake_input_scale: Incomplete
    def __init__(
        self, moe_config: mk.FusedMoEConfig, quant_config: FusedMoEQuantConfig
    ) -> None: ...
    @property
    def expects_unquantized_inputs(self) -> bool: ...
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
    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None: ...
