import types
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger

logger: Incomplete
HAS_TRITON: Incomplete
active_drivers: Incomplete
cuda_visible_devices: Incomplete
is_distributed_env: Incomplete

class TritonPlaceholder(types.ModuleType):
    __version__: str
    jit: Incomplete
    autotune: Incomplete
    heuristics: Incomplete
    Config: Incomplete
    language: Incomplete
    def __init__(self) -> None: ...

class TritonLanguagePlaceholder(types.ModuleType):
    constexpr: Incomplete
    dtype: Incomplete
    int64: Incomplete
    int32: Incomplete
    tensor: Incomplete
    exp: Incomplete
    log: Incomplete
    log2: Incomplete
    def __init__(self) -> None: ...
