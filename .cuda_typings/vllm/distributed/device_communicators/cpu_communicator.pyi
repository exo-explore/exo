import torch
from .base_device_communicator import DeviceCommunicatorBase as DeviceCommunicatorBase
from _typeshed import Incomplete
from torch.distributed import ProcessGroup as ProcessGroup
from typing import Any
from vllm.distributed.utils import pickle as pickle
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.platforms.interface import CpuArchEnum as CpuArchEnum

logger: Incomplete

class CpuCommunicator(DeviceCommunicatorBase):
    dist_module: Incomplete
    all2all_backend: str
    all2all_manager: Incomplete
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ) -> None: ...
    def all_reduce(self, input_): ...
    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None: ...
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor: ...
    def send_tensor_dict(
        self, tensor_dict: dict[str, torch.Tensor | Any], dst: int
    ) -> None: ...
    def recv_tensor_dict(self, src: int) -> dict[str, torch.Tensor | Any]: ...
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

class _CPUSHMDistributed:
    communicator: Incomplete
    group_name: Incomplete
    handle: Incomplete
    def __init__(self, communicator: CpuCommunicator) -> None: ...
    @staticmethod
    def make_group_name(communicator: CpuCommunicator) -> str: ...
    def all_reduce(
        self, input: torch.Tensor, group: ProcessGroup | None = None
    ) -> None: ...
    def gather(
        self,
        input: torch.Tensor,
        gather_list: list[torch.Tensor] | None,
        dst: int = -1,
        group: ProcessGroup | None = None,
    ) -> None: ...
    def all_gather_into_tensor(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: ProcessGroup | None = None,
    ) -> None: ...
    def send_tensor_dict(
        self, tensor_dict: dict[str, torch.Tensor | Any], dst: int
    ) -> None: ...
    def recv_tensor_dict(self, src: int) -> dict[str, torch.Tensor | Any]: ...
