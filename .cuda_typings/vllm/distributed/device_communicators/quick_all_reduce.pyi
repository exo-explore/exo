import torch
from _typeshed import Incomplete
from enum import Enum
from torch.distributed import ProcessGroup as ProcessGroup
from vllm.config import (
    get_current_vllm_config_or_none as get_current_vllm_config_or_none,
)
from vllm.distributed.parallel_state import in_the_same_node_as as in_the_same_node_as
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    cuda_device_count_stateless as cuda_device_count_stateless,
)

logger: Incomplete
quick_ar: bool

def is_weak_contiguous(inp: torch.Tensor): ...

class QuickReduceRegime(Enum):
    FP = 0
    INT8 = 1
    INT6 = 2
    INT4 = 3
    NONE = 4

MB: Incomplete

class QuickAllReduce:
    disabled: bool
    group: Incomplete
    rank: Incomplete
    world_size: Incomplete
    device: Incomplete
    fully_connected: Incomplete
    def __init__(
        self, group: ProcessGroup, device: int | str | torch.device
    ) -> None: ...
    use_fp16_kernels: Incomplete
    qr_quant_level: Incomplete
    qr_max_size: Incomplete
    def init_quick_all_reduce(self) -> None: ...
    def create_shared_buffer(self) -> None: ...
    def should_quick_allreduce(self, inp: torch.Tensor): ...
    def quick_all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None): ...
    def close(self) -> None: ...
    def __del__(self) -> None: ...
