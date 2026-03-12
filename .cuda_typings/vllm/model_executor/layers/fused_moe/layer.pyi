import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable
from enum import Enum
from typing import Literal, overload
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config import (
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.config.parallel import ExpertPlacementStrategy as ExpertPlacementStrategy
from vllm.distributed import (
    get_dp_group as get_dp_group,
    get_pcp_group as get_pcp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.distributed.eplb.eplb_state import (
    EplbLayerState as EplbLayerState,
    EplbState as EplbState,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
    RoutingMethodType as RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase as FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod as FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    init_aiter_topK_meta_data as init_aiter_topK_meta_data,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router as create_fused_moe_router,
)
from vllm.model_executor.layers.fused_moe.runner.default_moe_runner import (
    DefaultMoERunner as DefaultMoERunner,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod as UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.utils import (
    disable_inplace as disable_inplace,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.math_utils import round_up as round_up

logger: Incomplete

class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"

def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    num_fused_shared_experts: int = 0,
    return_expert_mask: bool = False,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]: ...
def determine_expert_placement_strategy(
    expert_placement_strategy: ExpertPlacementStrategy,
    moe_parallel_config: FusedMoEParallelConfig,
    num_expert_group: int | None,
    num_redundant_experts: int,
    enable_eplb: bool,
) -> ExpertPlacementStrategy: ...
def get_compressed_expert_map(expert_map: torch.Tensor) -> str: ...
def maybe_roundup_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    moe_parallel_config: FusedMoEParallelConfig,
    is_lora_enabled: bool,
    model_type: str | None,
    is_mxfp4_quant: bool,
) -> int: ...

class FusedMoE(CustomOp):
    params_dtype: Incomplete
    vllm_config: Incomplete
    is_sequence_parallel: Incomplete
    sp_size: Incomplete
    moe_parallel_config: FusedMoEParallelConfig
    global_num_experts: Incomplete
    logical_num_experts: Incomplete
    expert_mapping: Incomplete
    layer_name: Incomplete
    enable_eplb: Incomplete
    eplb_state: Incomplete
    expert_placement_strategy: ExpertPlacementStrategy
    rocm_aiter_fmoe_enabled: Incomplete
    aiter_fmoe_shared_expert_enabled: Incomplete
    num_fused_shared_experts: Incomplete
    local_num_experts: Incomplete
    top_k: Incomplete
    intermediate_size_per_partition: Incomplete
    reduce_results: Incomplete
    renormalize: Incomplete
    use_grouped_topk: Incomplete
    num_expert_group: Incomplete
    topk_group: Incomplete
    custom_routing_function: Incomplete
    scoring_func: Incomplete
    routed_scaling_factor: Incomplete
    e_score_correction_bias: Incomplete
    apply_router_weight_on_input: Incomplete
    activation: Incomplete
    router: Incomplete
    routing_method_type: RoutingMethodType
    model_type: Incomplete
    hidden_size: Incomplete
    moe_config: FusedMoEConfig
    quant_config: Incomplete
    quant_method: FusedMoEMethodBase
    base_quant_method: Incomplete
    use_overlapped: Incomplete
    runner: Incomplete
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
        pcp_size: int | None = None,
        prefix: str = "",
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        is_act_and_mul: bool = True,
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel: bool = False,
        expert_mapping: list[tuple[str, str, int, str]] | None = None,
        n_shared_experts: int | None = None,
        router_logits_dtype: torch.dtype | None = None,
        gate: torch.nn.Module | None = None,
        shared_experts: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
    ) -> None: ...
    def maybe_init_modular_kernel(self) -> None: ...
    @property
    def shared_experts(self) -> torch.nn.Module | None: ...
    @property
    def layer_id(self): ...
    @property
    def gate(self) -> torch.nn.Module | None: ...
    @property
    def tp_size(self): ...
    @property
    def ep_size(self): ...
    @property
    def tp_rank(self): ...
    @property
    def ep_rank(self): ...
    @property
    def use_ep(self): ...
    @property
    def is_internal_router(self) -> bool: ...
    @staticmethod
    def ensure_round_robin_expert_routing_tables(
        global_num_experts: int,
        ep_size: int,
        ep_rank: int,
        local_num_experts: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def update_expert_map(self) -> None: ...
    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[False],
    ) -> None: ...
    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[True],
    ) -> bool: ...
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]: ...
    def get_expert_weights(self) -> Iterable[torch.Tensor]: ...
    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None: ...
    def ensure_moe_quant_config_init(self) -> None: ...
    @property
    def moe_quant_config(self) -> FusedMoEQuantConfig | None: ...
    def must_reduce_shared_expert_outputs(self) -> bool: ...
    def maybe_all_reduce_tensor_model_parallel(
        self, final_hidden_states: torch.Tensor
    ): ...
    def forward_native(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    @property
    def expert_map(self) -> torch.Tensor | None: ...
    def forward_cuda(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    @classmethod
    def make_expert_params_mapping(
        cls,
        model: torch.nn.Module,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]: ...
    def extra_repr(self) -> str: ...
