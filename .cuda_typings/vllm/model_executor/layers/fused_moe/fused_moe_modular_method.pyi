import torch
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase as FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernel as FusedMoEKernel,
    FusedMoEPrepareAndFinalizeModular as FusedMoEPrepareAndFinalizeModular,
)

logger: Incomplete

class FusedMoEModularMethod(FusedMoEMethodBase, CustomOp):
    moe_quant_config: Incomplete
    moe_kernel: Incomplete
    disable_expert_map: Incomplete
    old_quant_method: Incomplete
    def __init__(
        self, old_quant_method: FusedMoEMethodBase, moe_kernel: FusedMoEKernel
    ) -> None: ...
    @staticmethod
    def make(
        moe_layer: torch.nn.Module,
        old_quant_method: FusedMoEMethodBase,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        shared_experts: torch.nn.Module | None,
        inplace: bool = False,
    ) -> FusedMoEModularMethod: ...
    @property
    def supports_eplb(self) -> bool: ...
    @property
    def method_name(self) -> str: ...
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
