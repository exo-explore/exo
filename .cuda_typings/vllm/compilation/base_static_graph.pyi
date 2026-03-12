from collections.abc import Callable as Callable
from typing import Any, Protocol
from vllm.config import CUDAGraphMode as CUDAGraphMode, VllmConfig as VllmConfig

class AbstractStaticGraphWrapper(Protocol):
    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        **kwargs: Any,
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
