import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from torch.distributed import Backend as Backend, ProcessGroup
from typing import Any, NamedTuple, Protocol
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase as DeviceCommunicatorBase,
)
from vllm.distributed.stateless_coordinator import (
    StatelessGroupCoordinator as StatelessGroupCoordinator,
)
from vllm.distributed.utils import StatelessProcessGroup as StatelessProcessGroup
from vllm.logger import init_logger as init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname
from vllm.utils.network_utils import (
    get_distributed_init_method as get_distributed_init_method,
)
from vllm.utils.system_utils import suppress_stdout as suppress_stdout
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream

class TensorMetadata(NamedTuple):
    device: Incomplete
    dtype: Incomplete
    size: Incomplete

class Handle(Protocol):
    def is_completed(self) -> bool: ...
    def wait(self) -> None: ...

def all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor: ...
def all_reduce_fake(tensor: torch.Tensor, group_name: str) -> torch.Tensor: ...
def reduce_scatter(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor: ...
def reduce_scatter_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor: ...
def all_gather(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor: ...
def all_gather_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor: ...
def patched_fused_scaled_matmul_reduce_scatter_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    bias: torch.Tensor | None = None,
    result_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor: ...
def patched_fused_scaled_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    bias: torch.Tensor | None = None,
    result_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor: ...

class GroupCoordinator:
    rank: int
    ranks: list[int]
    world_size: int
    local_rank: int
    rank_in_group: int
    cpu_group: ProcessGroup
    device_group: ProcessGroup
    device_communicator: DeviceCommunicatorBase | None
    mq_broadcaster: Any | None
    unique_name: Incomplete
    device: Incomplete
    use_device_communicator: Incomplete
    use_custom_op_call: Incomplete
    use_cpu_custom_send_recv: Incomplete
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
    ) -> None: ...
    def create_mq_broadcaster(
        self, writer_rank: int = 0, external_writer_handle=None, blocking: bool = True
    ): ...
    def create_single_reader_mq_broadcasters(
        self, reader_rank_in_group: int = 0, blocking: bool = False
    ): ...
    @property
    def first_rank(self): ...
    @property
    def last_rank(self): ...
    @property
    def is_first_rank(self): ...
    @property
    def is_last_rank(self): ...
    @property
    def next_rank(self): ...
    @property
    def prev_rank(self): ...
    @contextmanager
    def graph_capture(
        self, graph_capture_context: GraphCaptureContext | None = None
    ): ...
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor: ...
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor: ...
    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ): ...
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor: ...
    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor: ...
    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None: ...
    def broadcast(self, input_: torch.Tensor, src: int = 0): ...
    def broadcast_object(self, obj: Any | None = None, src: int = 0): ...
    def broadcast_object_list(
        self, obj_list: list[Any], src: int = 0, group: ProcessGroup | None = None
    ): ...
    def send_object(self, obj: Any, dst: int) -> None: ...
    def recv_object(self, src: int) -> Any: ...
    def broadcast_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any] | None = None,
        src: int = 0,
        group: ProcessGroup | None = None,
        metadata_group: ProcessGroup | None = None,
    ) -> dict[str, torch.Tensor | Any] | None: ...
    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
        all_gather_group: GroupCoordinator | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None: ...
    def isend_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
        all_gather_group: GroupCoordinator | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> list[Handle]: ...
    def recv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: GroupCoordinator | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None: ...
    def irecv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: GroupCoordinator | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor | Any] | None, list[Handle], list[Callable[[], None]]
    ]: ...
    def barrier(self) -> None: ...
    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None: ...
    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor: ...
    def destroy(self) -> None: ...
    def prepare_communication_buffer_for_model(self, model: torch.nn.Module): ...
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
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ): ...
    def combine(
        self, hidden_states, is_sequence_parallel: bool = False
    ) -> torch.Tensor: ...

def get_world_group() -> GroupCoordinator: ...
def get_inner_dp_world_group() -> GroupCoordinator: ...
def init_world_group(
    ranks: list[int], local_rank: int, backend: str
) -> GroupCoordinator: ...
def init_model_parallel_group(
    group_ranks: list[list[int]],
    local_rank: int,
    backend: str,
    use_message_queue_broadcaster: bool = False,
    group_name: str | None = None,
    use_device_communicator: bool = True,
) -> GroupCoordinator: ...
def get_tp_group() -> GroupCoordinator: ...
def get_dcp_group() -> GroupCoordinator: ...

get_context_model_parallel_group = get_dcp_group

def get_pp_group() -> GroupCoordinator: ...
def get_dp_group() -> GroupCoordinator: ...
def get_ep_group() -> GroupCoordinator: ...
def get_eplb_group() -> GroupCoordinator: ...
def get_pcp_group() -> GroupCoordinator: ...
@contextmanager
def graph_capture(device: torch.device): ...

logger: Incomplete

def set_custom_all_reduce(enable: bool): ...
def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: timedelta | None = None,
): ...
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None: ...
def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None: ...
def prepare_communication_buffer_for_model(model: torch.nn.Module): ...
def model_parallel_is_initialized(): ...
@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator): ...
def get_tensor_model_parallel_world_size() -> int: ...
def get_tensor_model_parallel_rank() -> int: ...
def get_decode_context_model_parallel_world_size() -> int: ...
def get_decode_context_model_parallel_rank() -> int: ...
def get_node_count() -> int: ...
def destroy_model_parallel() -> None: ...
def destroy_distributed_environment() -> None: ...
def cleanup_dist_env_and_memory(shutdown_ray: bool = False): ...
def in_the_same_node_as(
    pg: ProcessGroup | StatelessProcessGroup, source_rank: int = 0
) -> list[bool]: ...
def is_global_first_rank() -> bool: ...
def is_local_first_rank() -> bool: ...
