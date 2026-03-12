import torch
from .base_device_communicator import DeviceCommunicatorBase as DeviceCommunicatorBase
from _typeshed import Incomplete
from torch.distributed import ProcessGroup as ProcessGroup
from vllm.logger import init_logger as init_logger

logger: Incomplete

class XpuCommunicator(DeviceCommunicatorBase):
    all2all_manager: Incomplete
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ) -> None: ...
    def all_reduce(self, input_) -> torch.Tensor: ...
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1): ...
    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ): ...
    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ): ...
    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None: ...
    def broadcast(self, input_: torch.Tensor, src: int = 0) -> None: ...
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
