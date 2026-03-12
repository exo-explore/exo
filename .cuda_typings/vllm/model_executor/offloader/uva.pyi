import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Generator
from vllm.logger import init_logger as init_logger
from vllm.model_executor.offloader.base import BaseOffloader as BaseOffloader
from vllm.utils.mem_utils import format_gib as format_gib
from vllm.utils.platform_utils import (
    is_pin_memory_available as is_pin_memory_available,
    is_uva_available as is_uva_available,
)
from vllm.utils.torch_utils import (
    get_accelerator_view_from_cpu_tensor as get_accelerator_view_from_cpu_tensor,
)

logger: Incomplete

class UVAOffloader(BaseOffloader):
    cpu_offload_max_bytes: Incomplete
    cpu_offload_bytes: int
    cpu_offload_params: Incomplete
    pin_memory: Incomplete
    uva_offloading: Incomplete
    def __init__(
        self, cpu_offload_max_bytes: int, cpu_offload_params: set[str] | None = None
    ) -> None: ...
    def wrap_modules(
        self, modules_generator: Generator[nn.Module, None, None]
    ) -> list[nn.Module]: ...
