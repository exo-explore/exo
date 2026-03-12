import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm._custom_ops import (
    cpu_fused_moe as cpu_fused_moe,
    cpu_prepack_moe_weight as cpu_prepack_moe_weight,
)
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.quantization.utils.layer_utils import (
    replace_parameter as replace_parameter,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class SGLFusedMOE:
    def __init__(self, layer: torch.nn.Module) -> None: ...
    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: MoEActivation = ...,
    ) -> torch.Tensor: ...

class CPUFusedMOE:
    isa: Incomplete
    forward_method: Incomplete
    def __init__(self, layer: torch.nn.Module) -> None: ...
    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: MoEActivation = ...,
    ) -> torch.Tensor: ...
    def check_grouped_gemm(self, layer: torch.nn.Module) -> tuple[bool, str]: ...
    def init_moe_grouped_gemm(self, layer: torch.nn.Module) -> None: ...
    def init_moe_torch(self, layer: torch.nn.Module) -> None: ...
    def forward_grouped_gemm(
        self,
        layer: torch.nn.Module,
        input: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int = -1,
        skip_weighted: bool = False,
    ) -> torch.Tensor: ...
    def forward_torch(
        self,
        layer: torch.nn.Module,
        input: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int = -1,
        skip_weighted: bool = False,
    ) -> torch.Tensor: ...

def cpu_fused_moe_torch(
    layer_id: int,
    output: torch.Tensor,
    input: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    global_num_experts: int = -1,
    skip_weighted: bool = False,
) -> None: ...
