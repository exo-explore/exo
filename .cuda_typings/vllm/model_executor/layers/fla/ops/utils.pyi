import functools
import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from enum import Enum
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import triton as triton

logger: Incomplete
COMPILER_MODE: Incomplete
FLA_CI_ENV: Incomplete
FLA_GDN_FIX_BT: Incomplete
SUPPRESS_LEVEL: Incomplete

def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]: ...
def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]: ...
@functools.cache
def get_available_device() -> str: ...

device: Incomplete
device_torch_lib: Incomplete
device_platform: Incomplete
is_amd: Incomplete
is_intel: Incomplete
is_nvidia: Incomplete
is_intel_alchemist: Incomplete
is_nvidia_hopper: Incomplete
use_cuda_graph: Incomplete
is_gather_supported: Incomplete
is_tma_supported: Incomplete

def get_all_max_shared_mem(): ...

class Backend(Enum):
    ADA = 101376
    AMPERE = 166912
    HOPPER = 232448
    DEFAULT = 102400
    @classmethod
    def get_shared_memory(cls, arch: str) -> int: ...

@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool: ...
