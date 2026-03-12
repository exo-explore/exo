import contextlib
from _typeshed import Incomplete
from collections.abc import Generator
from vllm.config import CompilationMode as CompilationMode, VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger

logger: Incomplete
torch_compile_start_time: float

@contextlib.contextmanager
def monitor_torch_compile(
    vllm_config: VllmConfig, message: str = "torch.compile took %.2f s in total"
) -> Generator[None, None, None]: ...
@contextlib.contextmanager
def monitor_profiling_run() -> Generator[None, None, None]: ...

cudagraph_capturing_enabled: bool

def validate_cudagraph_capturing_enabled() -> None: ...
def set_cudagraph_capturing_enabled(enabled: bool) -> None: ...
