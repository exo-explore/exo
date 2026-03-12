import abc
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from compressed_tensors.quantization import QuantizationArgs
from enum import Enum
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

__all__ = [
    "CompressedTensorsMoEMethod",
    "CompressedTensorsW8A8Fp8MoEMethod",
    "CompressedTensorsW8A8Int8MoEMethod",
    "CompressedTensorsWNA16MarlinMoEMethod",
    "CompressedTensorsWNA16MoEMethod",
    "CompressedTensorsW4A4Nvfp4MoEMethod",
    "CompressedTensorsW4A8Int8MoEMethod",
]

class GPTQMarlinState(Enum):
    REPACK = ...
    READY = ...

class CompressedTensorsMoEMethod(FusedMoEMethodBase, metaclass=abc.ABCMeta):
    @staticmethod
    def get_moe_method(
        quant_config: CompressedTensorsConfig, layer: torch.nn.Module, layer_name: str
    ) -> FusedMoEMethodBase: ...

class CompressedTensorsW4A4Mxfp4MoEMethod(CompressedTensorsMoEMethod):
    group_size: int
    mxfp4_backend: Incomplete
    experts_cls: Incomplete
    def __init__(self, moe) -> None: ...
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
    moe_quant_config: Incomplete
    moe_kernel: Incomplete
    def process_weights_after_loading(self, layer: FusedMoE) -> None: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

class CompressedTensorsW4A4Nvfp4MoEMethod(CompressedTensorsMoEMethod):
    group_size: int
    use_global_sf: Incomplete
    def __init__(
        self, moe: FusedMoEConfig, layer_name: str | None = None, use_a16: bool = False
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
    moe_quant_config: Incomplete
    moe_kernel: Incomplete
    def process_weights_after_loading(self, layer: FusedMoE) -> None: ...
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None: ...
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig: ...
    def apply_monolithic(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

class CompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsMoEMethod):
    weight_quant: Incomplete
    input_quant: Incomplete
    weight_block_size: Incomplete
    block_quant: Incomplete
    static_input_scales: Incomplete
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
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
    moe_quant_config: Incomplete
    moe_kernel: Incomplete
    def process_weights_after_loading(self, layer: FusedMoE) -> None: ...
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None: ...
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig: ...
    def apply_monolithic(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    @property
    def supports_eplb(self) -> bool: ...

class CompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):
    weight_quant: Incomplete
    input_quant: Incomplete
    static_input_scales: Incomplete
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
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

class CompressedTensorsWNA16MarlinMoEMethod(CompressedTensorsMoEMethod):
    weight_quant: Incomplete
    input_quant: Incomplete
    num_bits: Incomplete
    packed_factor: Incomplete
    strategy: Incomplete
    group_size: Incomplete
    actorder: Incomplete
    quant_type: Incomplete
    marlin_input_dtype: Incomplete
    use_flashinfer_mxint4_moe: Incomplete
    kernel_backend: Incomplete
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ) -> None: ...
    def get_weight_shape(
        self,
        weight_name: str,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        num_groups_w2: int | None = None,
        num_groups_w13: int | None = None,
    ) -> tuple[int, int, int]: ...
    is_k_full: Incomplete
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
    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular: ...
    @property
    def is_monolithic(self) -> bool: ...
    def apply_monolithic(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

class CompressedTensorsWNA16MoEMethod(CompressedTensorsMoEMethod):
    weight_quant: Incomplete
    input_quant: Incomplete
    num_bits: Incomplete
    packed_factor: Incomplete
    strategy: Incomplete
    group_size: Incomplete
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
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
    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    @property
    def supports_eplb(self) -> bool: ...

class CompressedTensorsW4A8Int8MoEMethod(CompressedTensorsMoEMethod):
    has_bias: Incomplete
    weight_quant: Incomplete
    input_quant: Incomplete
    group_size: Incomplete
    static_input_scales: bool
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
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
    @property
    def is_monolithic(self) -> bool: ...
    def apply_monolithic(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor: ...

class CompressedTensorsW4A8Fp8MoEMethod(CompressedTensorsMoEMethod):
    weight_quant: Incomplete
    input_quant: Incomplete
    group_size: Incomplete
    num_bits: Incomplete
    packed_factor: Incomplete
    disable_expert_map: bool
    layer_name: Incomplete
    quant_fp8: Incomplete
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
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
    a_strides1_c_strides2: Incomplete
    a_strides2: Incomplete
    c_strides1: Incomplete
    s_strides1: Incomplete
    s_strides2: Incomplete
    def process_weights_after_loading(self, layer) -> None: ...
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None: ...
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None: ...
    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    @property
    def supports_eplb(self) -> bool: ...
