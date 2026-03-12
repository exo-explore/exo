import torch
from ..utils import StatelessProcessGroup as StatelessProcessGroup
from .base_device_communicator import DeviceCommunicatorBase as DeviceCommunicatorBase
from _typeshed import Incomplete
from torch.distributed import ProcessGroup as ProcessGroup
from vllm.distributed.device_communicators.all_reduce_utils import (
    should_nccl_symm_mem_allreduce as should_nccl_symm_mem_allreduce,
)
from vllm.distributed.device_communicators.pynccl import (
    register_nccl_symmetric_ops as register_nccl_symmetric_ops,
)
from vllm.distributed.device_communicators.pynccl_allocator import (
    is_symmetric_memory_enabled as is_symmetric_memory_enabled,
)
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform

logger: Incomplete

class CudaCommunicator(DeviceCommunicatorBase):
    use_custom_allreduce: Incomplete
    use_torch_symm_mem: Incomplete
    use_flashinfer_allreduce: Incomplete
    pynccl_comm: PyNcclCommunicator | None
    ca_comm: CustomAllreduce | None
    qr_comm: QuickAllReduce | None
    symm_mem_comm: SymmMemCommunicator | None
    fi_ar_comm: FlashInferAllReduce | None
    all2all_manager: Incomplete
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group: StatelessProcessGroup | None = None,
    ) -> None: ...
    def all_reduce(self, input_): ...
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1): ...
    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ): ...
    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None: ...
    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor: ...
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor: ...
    def destroy(self) -> None: ...
    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ): ...
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
