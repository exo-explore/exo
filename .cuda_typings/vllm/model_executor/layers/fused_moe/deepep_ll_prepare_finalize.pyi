import deep_ep
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm import envs as envs
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate as TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input as moe_kernel_quantize_input,
    normalize_batched_scales_shape as normalize_batched_scales_shape,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id as dbo_current_ubatch_id,
    dbo_enabled as dbo_enabled,
    dbo_maybe_run_recv_hook as dbo_maybe_run_recv_hook,
)

logger: Incomplete
DEEPEP_QUANT_BLOCK_SIZE: int
DEEPEP_QUANT_BLOCK_SHAPE: Incomplete

def dequant_fp8(
    expert_x_fp8: torch.Tensor, expert_x_scales: torch.Tensor
) -> torch.Tensor: ...

class DeepEPLLPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    SUPPORTED_HIDDEN_SIZES: Incomplete
    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int) -> int: ...
    buffer: Incomplete
    max_tokens_per_rank: Incomplete
    use_fp8_dispatch: Incomplete
    handles: list[tuple | None]
    num_dispatchers_: Incomplete
    global_to_physical: Incomplete
    physical_to_global: Incomplete
    local_expert_global_ids: Incomplete
    use_ue8m0_dispatch: bool
    def __init__(
        self,
        buffer: deep_ep.Buffer,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        global_to_physical: torch.Tensor | None = None,
        physical_to_global: torch.Tensor | None = None,
        local_expert_global_ids: torch.Tensor | None = None,
    ) -> None: ...
    def post_init_setup(self, fused_experts: mk.FusedMoEExperts): ...
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
    ) -> tuple[Callable, mk.ReceiverType]: ...
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
    ) -> tuple[Callable, Callable]: ...
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None: ...
