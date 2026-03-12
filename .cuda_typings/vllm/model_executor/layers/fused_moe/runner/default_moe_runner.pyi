import torch
from _typeshed import Incomplete
from vllm.distributed import (
    get_ep_group as get_ep_group,
    get_pcp_group as get_pcp_group,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
    is_forward_context_available as is_forward_context_available,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig as FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase as FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter as FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner as MoERunner,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.torch_utils import (
    HAS_OPAQUE_TYPE as HAS_OPAQUE_TYPE,
    ModuleName as ModuleName,
    aux_stream as aux_stream,
    current_stream as current_stream,
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id as dbo_current_ubatch_id

logger: Incomplete

def get_layer_from_name(layer_name: str) -> torch.nn.Module: ...

class DefaultMoERunner(MoERunner):
    moe_config: Incomplete
    router: Incomplete
    routed_input_transform: Incomplete
    gate: Incomplete
    shared_experts: Incomplete
    quant_method: Incomplete
    reduce_results: Incomplete
    enable_dbo: Incomplete
    shared_experts_stream: Incomplete
    layer_name: Incomplete
    moe_forward: Incomplete
    batched_hidden_states: torch.Tensor | None
    batched_router_logits: torch.Tensor | None
    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_input_transform: torch.nn.Module | None,
        gate: torch.nn.Module | None,
        shared_experts: torch.nn.Module | None,
        quant_method: FusedMoEMethodBase,
        reduce_results: bool,
        enable_dbo: bool,
    ) -> None: ...
    @property
    def use_dp_chunking(self) -> bool: ...
    def ensure_dp_chunking_init(self) -> None: ...
    def must_reduce_shared_expert_outputs(self) -> bool: ...
    def maybe_all_reduce_tensor_model_parallel(
        self, final_hidden_states: torch.Tensor
    ): ...
    def apply_routed_input_transform(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...
    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def forward_impl_chunked(
        self,
        layer: torch.nn.Module,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
        full_shared_input: torch.Tensor | None,
        has_separate_shared_experts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
