import torch
from .base_device_communicator import (
    All2AllManagerBase as All2AllManagerBase,
    Cache as Cache,
)
from _typeshed import Incomplete
from vllm.distributed import get_dp_group as get_dp_group, get_ep_group as get_ep_group
from vllm.forward_context import get_forward_context as get_forward_context
from vllm.logger import init_logger as init_logger
from vllm.utils.flashinfer import has_flashinfer_all2all as has_flashinfer_all2all
from vllm.utils.import_utils import has_deep_ep as has_deep_ep, has_mori as has_mori

logger: Incomplete

class NaiveAll2AllManager(All2AllManagerBase):
    def __init__(self, cpu_group, tcp_store_group=None) -> None: ...
    def naive_multicast(
        self,
        x: torch.Tensor,
        cu_tokens_across_sp_cpu: torch.Tensor,
        is_sequence_parallel: bool,
    ) -> torch.Tensor: ...
    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor: ...
    def destroy(self) -> None: ...

class AgRsAll2AllManager(All2AllManagerBase):
    def __init__(self, cpu_group, tcp_store_group=None) -> None: ...
    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ): ...
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ): ...
    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor: ...
    def destroy(self) -> None: ...

class DeepEPAll2AllManagerBase(All2AllManagerBase):
    handle_cache: Incomplete
    num_sms: int
    def __init__(self, cpu_group, tcp_store_group=None) -> None: ...
    def get_handle(self, kwargs) -> None: ...
    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ): ...
    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor: ...
    def destroy(self) -> None: ...

class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    def __init__(self, cpu_group, tcp_store_group=None) -> None: ...
    def get_handle(self, kwargs): ...
    def set_num_sms(self, num_sms: int): ...

class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    def __init__(self, cpu_group, tcp_store_group=None) -> None: ...
    def get_handle(self, kwargs): ...
    def max_sms_used(self) -> int | None: ...

class FlashInferAllToAllManager(All2AllManagerBase):
    rank: int
    world_size: int
    initialized: bool
    alltoall_info: Incomplete
    def __init__(self, cpu_group, tcp_store_group=None) -> None: ...
    mapping: Incomplete
    workspace_tensor: Incomplete
    prepare_workspace_tensor: Incomplete
    gpus_per_node: Incomplete
    def initialize(self, world_size: int, rank: int, gpus_per_node: int): ...
    def ensure_alltoall_workspace_initialized(self): ...
    def get_handle(self, kwargs): ...
    def cleanup(self) -> None: ...

class MoriAll2AllManager(All2AllManagerBase):
    handle_cache: Incomplete
    def __init__(self, cpu_group) -> None: ...
    def get_handle(self, kwargs): ...
