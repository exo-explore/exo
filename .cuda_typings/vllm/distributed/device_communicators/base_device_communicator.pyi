import torch
from _typeshed import Incomplete
from torch.distributed import ProcessGroup as ProcessGroup

class Cache:
    def __init__(self) -> None: ...
    def get_or_create(self, kwargs, func): ...

class All2AllManagerBase:
    rank: int
    world_size: int
    cpu_group: Incomplete
    tcp_store_group: Incomplete
    dp_group: Incomplete
    tp_group: Incomplete
    dp_rank: Incomplete
    dp_world_size: Incomplete
    internode: Incomplete
    def __init__(self, cpu_group, tcp_store_group=None) -> None: ...
    def get_handle(self, kwargs) -> None: ...
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
    def set_num_sms(self, num_sms: int): ...
    def max_sms_used(self) -> int | None: ...
    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ): ...
    def destroy(self) -> None: ...

class DeviceCommunicatorBase:
    device: Incomplete
    cpu_group: Incomplete
    device_group: Incomplete
    unique_name: Incomplete
    rank: Incomplete
    world_size: Incomplete
    ranks: Incomplete
    global_rank: Incomplete
    global_world_size: Incomplete
    rank_in_group: Incomplete
    is_ep_communicator: Incomplete
    use_all2all: Incomplete
    all2all_backend: Incomplete
    all2all_manager: All2AllManagerBase | None
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
    ) -> None: ...
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor: ...
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor: ...
    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]: ...
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor: ...
    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor: ...
    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None: ...
    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None: ...
    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor: ...
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor: ...
    def destroy(self) -> None: ...
    def prepare_communication_buffer_for_model(
        self, model: torch.nn.Module
    ) -> None: ...
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
    def batch_isend_irecv(self, p2p_ops: list): ...
