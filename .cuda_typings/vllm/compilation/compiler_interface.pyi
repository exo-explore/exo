import contextlib
import torch.fx as fx
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from typing import Any, Literal
from vllm.compilation.counter import compilation_counter as compilation_counter
from vllm.config import VllmConfig as VllmConfig
from vllm.config.utils import Range as Range
from vllm.logger import init_logger as init_logger
from vllm.utils.hashing import safe_hash as safe_hash
from vllm.utils.torch_utils import is_torch_equal_or_newer as is_torch_equal_or_newer

logger: Incomplete

class CompilerInterface:
    name: str
    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None: ...
    def compute_hash(self, vllm_config: VllmConfig) -> str: ...
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]: ...
    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable[..., Any]: ...

class AlwaysHitShapeEnv:
    guards: list[Any]
    def __init__(self) -> None: ...
    def evaluate_guards_expression(
        self, *args: Any, **kwargs: Any
    ) -> Literal[True]: ...
    def get_pruned_guards(self, *args: Any, **kwargs: Any) -> list[Any]: ...
    def produce_guards_expression(self, *args: Any, **kwargs: Any) -> Literal[""]: ...

def get_inductor_factors() -> list[Any]: ...
def is_compile_cache_enabled(
    vllm_additional_inductor_config: dict[str, Any],
) -> bool: ...

class InductorStandaloneAdaptor(CompilerInterface):
    name: str
    save_format: Incomplete
    def __init__(self, save_format: Literal["binary", "unpacked"]) -> None: ...
    def compute_hash(self, vllm_config: VllmConfig) -> str: ...
    cache_dir: Incomplete
    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None: ...
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]: ...
    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable[..., Any]: ...

class InductorAdaptor(CompilerInterface):
    name: str
    def compute_hash(self, vllm_config: VllmConfig) -> str: ...
    cache_dir: Incomplete
    prefix: Incomplete
    base_cache_dir: Incomplete
    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None: ...
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]: ...
    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable[..., Any]: ...
    def metrics_context(self) -> contextlib.AbstractContextManager[Any]: ...

def set_inductor_config(config: dict[str, Any], compile_range: Range) -> None: ...
def set_functorch_config() -> None: ...

class EagerAdaptor(CompilerInterface):
    name: str
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]: ...
