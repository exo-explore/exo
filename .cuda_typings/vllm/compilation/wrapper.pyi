import abc
import torch
from _typeshed import Incomplete
from abc import abstractmethod
from collections.abc import Callable as Callable
from types import CodeType
from typing import Any, ParamSpec, TypeVar
from vllm.config import (
    CUDAGraphMode as CUDAGraphMode,
    CompilationMode as CompilationMode,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.config.compilation import DynamicShapesType as DynamicShapesType
from vllm.logger import init_logger as init_logger
from vllm.utils.nvtx_pytorch_hooks import (
    layerwise_nvtx_marker_context as layerwise_nvtx_marker_context,
)

logger: Incomplete
R = TypeVar("R")
P = ParamSpec("P")

class TorchCompileWithNoGuardsWrapper(metaclass=abc.ABCMeta):
    def check_invariants_and_forward(self, *args: Any, **kwargs: Any) -> Any: ...
    compiled: bool
    vllm_config: Incomplete
    layerwise_nvtx_tracing_enabled: Incomplete
    first_compile: bool
    evaluate_guards: Incomplete
    def __init__(self) -> None: ...
    def aot_compile(self, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def original_code_object(self) -> CodeType: ...
    def bytecode_hook(self, old_code: CodeType, new_code: CodeType) -> None: ...

def reset_compile_wrapper(model: torch.nn.Module) -> None: ...
