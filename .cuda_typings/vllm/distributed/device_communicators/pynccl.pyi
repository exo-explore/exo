import torch
from _typeshed import Incomplete
from torch.distributed import ProcessGroup as ProcessGroup, ReduceOp
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary as NCCLLibrary,
    buffer_type as buffer_type,
    cudaStream_t as cudaStream_t,
    ncclComm_t as ncclComm_t,
    ncclDataTypeEnum as ncclDataTypeEnum,
    ncclRedOpTypeEnum as ncclRedOpTypeEnum,
    ncclUniqueId as ncclUniqueId,
)
from vllm.distributed.utils import StatelessProcessGroup as StatelessProcessGroup
from vllm.logger import init_logger as init_logger
from vllm.utils.torch_utils import current_stream as current_stream

logger: Incomplete

def register_nccl_symmetric_ops(pynccl_comm): ...

class PyNcclCommunicator:
    rank: Incomplete
    world_size: Incomplete
    group: Incomplete
    available: bool
    disabled: bool
    nccl: Incomplete
    nccl_version: Incomplete
    unique_id: Incomplete
    device: Incomplete
    comm: ncclComm_t
    def __init__(
        self,
        group: ProcessGroup | StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
    ) -> None: ...
    def all_reduce(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor = None,
        op: ReduceOp = ...,
        stream=None,
    ) -> torch.Tensor: ...
    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None
    ): ...
    def all_gatherv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        stream=None,
    ): ...
    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ...,
        stream=None,
    ): ...
    def reduce_scatterv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        op: ReduceOp = ...,
        stream=None,
    ): ...
    def send(self, tensor: torch.Tensor, dst: int, stream=None): ...
    def recv(self, tensor: torch.Tensor, src: int, stream=None): ...
    def broadcast(self, tensor: torch.Tensor, src: int, stream=None): ...
    def group_start(self) -> None: ...
    def group_end(self) -> None: ...
    def register_comm_window(self, tensor: torch.Tensor): ...
    def register_comm_window_raw(self, ptr: int, size: int): ...
    def deregister_comm_window(self, window): ...
    def batch_isend_irecv(self, p2p_ops: list, stream=None): ...
