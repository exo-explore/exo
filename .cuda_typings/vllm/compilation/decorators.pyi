import contextlib
from .monitor import (
    monitor_profiling_run as monitor_profiling_run,
    monitor_torch_compile as monitor_torch_compile,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Generator
from typing import Any, overload
from vllm.compilation.counter import compilation_counter as compilation_counter
from vllm.compilation.wrapper import (
    TorchCompileWithNoGuardsWrapper as TorchCompileWithNoGuardsWrapper,
)
from vllm.config import (
    CompilationMode as CompilationMode,
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
    set_current_vllm_config as set_current_vllm_config,
)
from vllm.config.compilation import DynamicShapesType as DynamicShapesType
from vllm.forward_context import (
    get_forward_context as get_forward_context,
    is_forward_context_available as is_forward_context_available,
)
from vllm.logger import init_logger as init_logger
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname
from vllm.utils.torch_utils import is_torch_equal_or_newer as is_torch_equal_or_newer

SourceInfo = Any
logger: Incomplete
IGNORE_COMPILE_KEY: str

def should_torch_compile_mm_encoder(vllm_config: VllmConfig) -> bool: ...
def ignore_torch_compile(cls) -> type[_T]: ...
@overload
def support_torch_compile(
    *, enable_if: Callable[[VllmConfig], bool] | None = None
) -> Callable[[type[_T]], type[_T]]: ...
@overload
def support_torch_compile(
    *, dynamic_arg_dims: dict[str, int | list[int]] | None
) -> Callable[[type[_T]], type[_T]]: ...
@overload
def support_torch_compile(
    *, mark_unbacked_dims: dict[str, int | list[int]] | None
) -> Callable[[type[_T]], type[_T]]: ...
@overload
def support_torch_compile(
    *,
    dynamic_arg_dims: dict[str, int | list[int]] | None,
    mark_unbacked_dims: dict[str, int | list[int]] | None,
) -> Callable[[type[_T]], type[_T]]: ...
@overload
def support_torch_compile(cls) -> type[_T]: ...
@contextlib.contextmanager
def maybe_use_cudagraph_partition_wrapper(
    vllm_config: VllmConfig,
) -> Generator[None, None, None]: ...
