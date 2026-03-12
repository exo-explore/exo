import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import (
    CutlassExpertsFp8 as CutlassExpertsFp8,
)
from vllm.model_executor.layers.fused_moe.fallback import (
    FallbackExperts as FallbackExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts as TritonExperts,
)
from vllm.platforms import current_platform as current_platform

class TritonOrCutlassExperts(FallbackExperts):
    is_sm100: Incomplete
    def __init__(
        self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig
    ) -> None: ...
    @staticmethod
    def get_clses() -> tuple[
        type[mk.FusedMoEExpertsModular], type[mk.FusedMoEExpertsModular]
    ]: ...
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
