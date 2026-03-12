import torch
from _typeshed import Incomplete
from torch.distributed import Backend as Backend, ProcessGroup as ProcessGroup
from typing import Any
from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator as CudaCommunicator,
)
from vllm.distributed.parallel_state import (
    GroupCoordinator as GroupCoordinator,
    TensorMetadata as TensorMetadata,
)
from vllm.distributed.utils import (
    StatelessProcessGroup as StatelessProcessGroup,
    stateless_destroy_torch_distributed_process_group as stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group as stateless_init_torch_distributed_process_group,
)
from vllm.logger import init_logger as init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname

logger: Incomplete

class StatelessGroupCoordinator(GroupCoordinator):
    unique_name: Incomplete
    rank: Incomplete
    local_rank: Incomplete
    backend: Incomplete
    ranks: Incomplete
    world_size: Incomplete
    rank_in_group: Incomplete
    cpu_group: Incomplete
    device_group: Incomplete
    tcp_store_group: Incomplete
    device: Incomplete
    use_device_communicator: Incomplete
    device_communicator: Incomplete
    mq_broadcaster: Incomplete
    use_custom_op_call: Incomplete
    use_cpu_custom_send_recv: bool
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
        host: str = "127.0.0.1",
        group_ports: list[list[int]] | None = None,
        global_rank: int = 0,
        global_world_size: int = 1,
    ) -> None: ...
    def destroy(self) -> None: ...
    def size(self) -> int: ...
    def broadcast(self, input_: torch.Tensor, src: int = 0): ...
    def broadcast_object(self, obj=None, src: int = 0): ...
    def broadcast_object_list(
        self, obj_list: list[Any], src: int = 0, group: ProcessGroup | None = None
    ): ...
    def broadcast_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any] | None = None,
        src: int = 0,
        group: ProcessGroup | None = None,
        metadata_group: ProcessGroup | None = None,
    ) -> dict[str, torch.Tensor | Any] | None: ...
    def send_object(self, obj, dst: int) -> None: ...
    def recv_object(self, src: int): ...
    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
        all_gather_group: GroupCoordinator | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None: ...
    def recv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: GroupCoordinator | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None: ...
    def barrier(self) -> None: ...
    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None: ...
