import deep_ep
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from collections.abc import Callable as Callable
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
from vllm.utils.math_utils import round_up as round_up
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id as dbo_current_ubatch_id,
    dbo_enabled as dbo_enabled,
    dbo_get_previous_event as dbo_get_previous_event,
    dbo_switch_to_comm as dbo_switch_to_comm,
    dbo_switch_to_compute as dbo_switch_to_compute,
    dbo_switch_to_compute_sync as dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_comm_to_compute as dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm as dbo_yield_and_switch_from_compute_to_comm,
)

class DeepEPHTPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    @staticmethod
    def maybe_roundup_layer_hidden_size(
        hidden_size: int, dtype: torch.dtype
    ) -> int: ...
    buffer: Incomplete
    num_dispatchers_: Incomplete
    dp_size: Incomplete
    rank_expert_offset: Incomplete
    async_prepare: bool
    handles: Incomplete
    available_rank_configs: Incomplete
    def __init__(
        self,
        buffer: deep_ep.Buffer,
        num_dispatchers: int,
        dp_size: int,
        rank_expert_offset: int,
    ) -> None: ...
    def num_dispatchers(self) -> int: ...
    def output_is_reduced(self) -> bool: ...
    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat: ...
    def max_num_tokens_per_rank(self) -> int | None: ...
    def topk_indices_dtype(self) -> torch.dtype | None: ...
    def supports_async(self) -> bool: ...
    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType: ...
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
    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable: ...
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None: ...
