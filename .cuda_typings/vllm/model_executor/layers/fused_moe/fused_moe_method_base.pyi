import abc
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from abc import abstractmethod
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular as FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular as FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase as QuantizeMethodBase,
)

logger: Incomplete

class FusedMoEMethodBase(QuantizeMethodBase, metaclass=abc.ABCMeta):
    moe: FusedMoEConfig
    moe_quant_config: FusedMoEQuantConfig | None
    moe_kernel: mk.FusedMoEKernel | None
    def __init__(self, moe: FusedMoEConfig) -> None: ...
    @property
    def supports_internal_mk(self) -> bool: ...
    @property
    def mk_owns_shared_expert(self) -> bool: ...
    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def uses_weight_scale_2_pattern(self) -> bool: ...
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None: ...
    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> FusedMoEExpertsModular: ...
    @abstractmethod
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None: ...
    @property
    def topk_indices_dtype(self) -> torch.dtype | None: ...
    @property
    def supports_eplb(self) -> bool: ...
    @property
    def method_name(self) -> str: ...
    @property
    def is_monolithic(self) -> bool: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def apply_monolithic(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
