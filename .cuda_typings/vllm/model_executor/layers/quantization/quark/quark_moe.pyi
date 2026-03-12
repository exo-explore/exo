import abc
import torch
from _typeshed import Incomplete
from typing import Any
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Backend

__all__ = ["QuarkMoEMethod", "QuarkOCP_MX_MoEMethod", "QuarkOCP_MX_MoEMethod_OSS"]

class QuarkMoEMethod(FusedMoEMethodBase, metaclass=abc.ABCMeta):
    has_bias: Incomplete
    def __init__(self, moe: FusedMoEConfig) -> None: ...
    @staticmethod
    def get_moe_method(
        quant_config: QuarkConfig, module: torch.nn.Module, layer_name: str
    ) -> QuarkMoEMethod: ...

class QuarkW8A8Fp8MoEMethod(QuarkMoEMethod):
    weight_quant: Incomplete
    input_quant: Incomplete
    weight_qscheme: Incomplete
    input_qscheme: Incomplete
    weight_dtype: Incomplete
    input_dtype: Incomplete
    act_quant_group_shape: Incomplete
    static_input_scales: Incomplete
    use_marlin: Incomplete
    rocm_aiter_moe_enabled: Incomplete
    model_type: Incomplete
    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        moe: FusedMoEConfig,
    ) -> None: ...
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
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

class QuarkW4A8Fp8MoEMethod(QuarkMoEMethod):
    weight_quant: Incomplete
    input_quant: Incomplete
    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        moe: FusedMoEConfig,
    ) -> None: ...
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def get_fused_moe_quant_config(self, layer): ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

class QuarkOCP_MX_MoEMethod(QuarkMoEMethod):
    weight_quant: Incomplete
    input_quant: Incomplete
    weight_dtype: Incomplete
    input_dtype: Incomplete
    fp4_dtype: Incomplete
    ocp_mx_scheme: Incomplete
    mxfp4_backend: Mxfp4Backend | None
    static_input_scales: Incomplete
    use_rocm_aiter_moe: Incomplete
    model_type: Incomplete
    emulate: Incomplete
    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any] | None,
        moe: FusedMoEConfig,
    ) -> None: ...
    def get_packed_dim(self, dim: int, quant_dtype: str): ...
    intermediate_size_per_partition: Incomplete
    unpadded_hidden_size: Incomplete
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def process_weights_after_loading(self, layer) -> None: ...
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

class QuarkOCP_MX_MoEMethod_OSS(QuarkOCP_MX_MoEMethod):
    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        moe: FusedMoEConfig,
    ) -> None: ...
    w13_weight_triton_tensor: Incomplete
    w2_weight_triton_tensor: Incomplete
    w13_precision_config: Incomplete
    w2_precision_config: Incomplete
    def process_weights_after_loading(self, layer) -> None: ...
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None: ...
    @property
    def is_monolithic(self) -> bool: ...
    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        expert_map: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
