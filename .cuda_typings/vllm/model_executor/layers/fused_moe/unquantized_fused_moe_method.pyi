import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from .fused_batched_moe import BatchedTritonExperts as BatchedTritonExperts
from .fused_moe import TritonExperts as TritonExperts
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from torch.nn import Module as Module
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.logger import init_logger as init_logger
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG as FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
    biased_moe_quant_config as biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase as FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat as FusedMoEActivationFormat,
    FusedMoEExpertsModular as FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular as FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend as UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format as convert_to_unquantized_kernel_format,
    make_unquantized_moe_kernel as make_unquantized_moe_kernel,
    select_unquantized_moe_backend as select_unquantized_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    convert_moe_weights_to_flashinfer_trtllm_block_layout as convert_moe_weights_to_flashinfer_trtllm_block_layout,
)
from vllm.model_executor.utils import (
    replace_parameter as replace_parameter,
    set_weight_attrs as set_weight_attrs,
)
from vllm.platforms import current_platform as current_platform
from vllm.platforms.interface import CpuArchEnum as CpuArchEnum

logger: Incomplete

class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    unquantized_backend: Incomplete
    rocm_aiter_moe_enabled: Incomplete
    kernel: mk.FusedMoEKernel | None
    apply_monolithic: Callable
    def __init__(self, moe: FusedMoEConfig) -> None: ...
    def forward_native(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    @property
    def is_monolithic(self) -> bool: ...
    @property
    def supports_eplb(self) -> bool: ...
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None: ...
    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> FusedMoEExpertsModular: ...
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    cpu_fused_moe: Callable
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig: ...
    def forward_cuda(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def forward_monolithic_cuda(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def forward_monolithic_cpu(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
