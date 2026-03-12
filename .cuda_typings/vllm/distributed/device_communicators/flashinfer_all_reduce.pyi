import torch
from _typeshed import Incomplete
from torch.distributed import ProcessGroup as ProcessGroup
from vllm.config.compilation import PassConfig as PassConfig
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform

logger: Incomplete
fi_ar_available: bool

def get_fi_ar_workspace(): ...
def get_fi_ar_quant_workspace(): ...
def initialize_fi_ar_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
) -> None: ...
def initialize_fi_ar_quant_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
) -> None: ...
def destroy_fi_ar_workspace() -> None: ...

class FlashInferAllReduce:
    disabled: bool
    group: Incomplete
    world_size: Incomplete
    rank: Incomplete
    device: Incomplete
    max_workspace_size: Incomplete
    max_num_tokens: int
    def __init__(
        self, group: ProcessGroup, device: int | str | torch.device
    ) -> None: ...
    def should_use_fi_ar(self, input_tensor: torch.Tensor) -> bool: ...
    def all_reduce(self, input_tensor: torch.Tensor) -> torch.Tensor: ...
    def destroy(self) -> None: ...
