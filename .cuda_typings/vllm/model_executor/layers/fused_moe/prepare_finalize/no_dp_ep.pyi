import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous as TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate as TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input as moe_kernel_quantize_input,
)

class MoEPrepareAndFinalizeNoDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat: ...
    def max_num_tokens_per_rank(self) -> int | None: ...
    def topk_indices_dtype(self) -> torch.dtype | None: ...
    def num_dispatchers(self) -> int: ...
    def output_is_reduced(self) -> bool: ...
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType: ...
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None: ...

class MoEPrepareAndFinalizeNoDPEPMonolithic(mk.FusedMoEPrepareAndFinalizeMonolithic):
    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat: ...
    def max_num_tokens_per_rank(self) -> int | None: ...
    def topk_indices_dtype(self) -> torch.dtype | None: ...
    def num_dispatchers(self) -> int: ...
    def output_is_reduced(self) -> bool: ...
    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareMonolithicResultType: ...
    def finalize(self, fused_expert_output: torch.Tensor) -> torch.Tensor: ...

def make_moe_prepare_and_finalize_no_dp_ep(
    use_monolithic: bool,
) -> MoEPrepareAndFinalizeNoDPEPModular | MoEPrepareAndFinalizeNoDPEPMonolithic: ...
