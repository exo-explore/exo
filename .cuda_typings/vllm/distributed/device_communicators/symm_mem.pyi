import torch
from _typeshed import Incomplete
from torch.distributed import ProcessGroup as ProcessGroup
from vllm.distributed.device_communicators.all_reduce_utils import (
    SYMM_MEM_ALL_REDUCE_MAX_SIZES as SYMM_MEM_ALL_REDUCE_MAX_SIZES,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.platforms import current_platform as current_platform

symm_mem_available: bool
logger: Incomplete

class SymmMemCommunicator:
    disabled: bool
    dtype: Incomplete
    device: Incomplete
    group: Incomplete
    world_size: Incomplete
    device_capability: Incomplete
    max_size: Incomplete
    buffer: Incomplete
    force_multimem: Incomplete
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        force_multimem: bool | None = None,
        max_size_override: int | None = None,
    ) -> None: ...
    def should_use_symm_mem(self, inp: torch.Tensor): ...
    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor | None: ...
