import numpy as np
import threading
import torch
from .async_worker import start_async_worker as start_async_worker
from .policy import (
    AbstractEplbPolicy as AbstractEplbPolicy,
    DefaultEplbPolicy as DefaultEplbPolicy,
    EPLB_POLICIES as EPLB_POLICIES,
)
from .rebalance_execute import (
    RecvMetadata as RecvMetadata,
    move_from_buffer as move_from_buffer,
    rearrange_expert_weights_inplace as rearrange_expert_weights_inplace,
)
from _typeshed import Incomplete
from collections.abc import Sequence
from dataclasses import dataclass
from torch.distributed import ProcessGroup
from vllm.config import ModelConfig as ModelConfig, ParallelConfig as ParallelConfig
from vllm.distributed.parallel_state import (
    get_ep_group as get_ep_group,
    get_node_count as get_node_count,
    in_the_same_node_as as in_the_same_node_as,
)
from vllm.distributed.stateless_coordinator import (
    StatelessGroupCoordinator as StatelessGroupCoordinator,
)
from vllm.distributed.utils import StatelessProcessGroup as StatelessProcessGroup
from vllm.logger import init_logger as init_logger
from vllm.model_executor.models.interfaces import MixtureOfExperts as MixtureOfExperts

logger: Incomplete

@dataclass
class EplbStats:
    global_expert_load_window: torch.Tensor
    num_replicas: int
    num_groups: int
    num_nodes: int
    num_gpus: int

@dataclass
class EplbModelState:
    physical_to_logical_map: torch.Tensor
    logical_to_physical_map: torch.Tensor
    logical_replica_count: torch.Tensor
    expert_load_pass: torch.Tensor
    expert_load_window: torch.Tensor
    model_name: str
    model: MixtureOfExperts
    expert_buffer: list[torch.Tensor]
    buffer_lock: threading.Lock
    buffer_ready_event: torch.cuda.Event | None
    buffer_consumed_event: torch.cuda.Event | None
    window_ready_event: torch.cuda.Event | None
    ep_buffer_ready: int
    layer_to_transfer: int
    rebalanced: bool
    pending_global_ready_check: bool
    eplb_stats: EplbStats | None
    is_unchanged: np.ndarray
    is_received_locally: np.ndarray
    recv_metadata: RecvMetadata
    cuda_device_index: int | None
    new_physical_to_logical_map: torch.Tensor | None = ...
    new_logical_to_physical_map: torch.Tensor | None = ...
    new_logical_replica_count: torch.Tensor | None = ...

class EplbState:
    parallel_config: Incomplete
    device: Incomplete
    model_states: dict[str, EplbModelState]
    policy: type[AbstractEplbPolicy]
    expert_load_window_step: int
    expert_load_window_size: int
    expert_rearrangement_step: int
    expert_rearrangement_step_interval: int
    is_async: bool
    rearrange_event: Incomplete
    async_worker: threading.Thread | None
    cuda_device_index: int | None
    num_valid_physical_experts: int
    def __init__(
        self, parallel_config: ParallelConfig, device: torch.device
    ) -> None: ...
    @staticmethod
    def build_initial_global_physical_to_logical_map(
        num_routed_experts: int, num_redundant_experts: int
    ) -> Sequence[int]: ...
    def validate_ep_configuration(self, new_model: MixtureOfExperts): ...
    def add_model(self, model: MixtureOfExperts, model_config: ModelConfig): ...
    def step(
        self, is_dummy: bool = False, is_profile: bool = False, log_stats: bool = False
    ) -> None: ...
    def rearrange(
        self, is_profile: bool = False, rank_mapping: dict[int, int] | None = None
    ) -> torch.Tensor | None: ...
    def start_async_loop(
        self, rank_mapping: dict[int, int] | None = None, is_profile: bool = False
    ): ...
    def move_to_workspace(
        self,
        model_state: EplbModelState,
        ep_group: ProcessGroup,
        is_profile: bool = False,
    ): ...
    def post_eplb(
        self, model_state: EplbModelState, is_profile: bool = False
    ) -> None: ...
    @classmethod
    def from_mapping(
        cls,
        model: MixtureOfExperts,
        model_config: ModelConfig,
        device: torch.device,
        parallel_config: ParallelConfig,
        expanded_physical_to_logical: torch.Tensor,
        num_valid_physical_experts: int,
    ) -> EplbState: ...

@dataclass
class EplbLayerState:
    expert_load_view: torch.Tensor | None = ...
    logical_to_physical_map: torch.Tensor | None = ...
    logical_replica_count: torch.Tensor | None = ...
