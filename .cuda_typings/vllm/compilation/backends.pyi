import dataclasses
import torch
import torch.fx as fx
from .compiler_interface import (
    CompilerInterface as CompilerInterface,
    EagerAdaptor as EagerAdaptor,
    InductorAdaptor as InductorAdaptor,
    InductorStandaloneAdaptor as InductorStandaloneAdaptor,
    is_compile_cache_enabled as is_compile_cache_enabled,
)
from .counter import compilation_counter as compilation_counter
from .partition_rules import (
    inductor_partition_rule_context as inductor_partition_rule_context,
    should_split as should_split,
)
from .passes.inductor_pass import (
    InductorPass as InductorPass,
    pass_context as pass_context,
)
from .passes.pass_manager import PostGradPassManager as PostGradPassManager
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any
from vllm.config import (
    CUDAGraphMode as CUDAGraphMode,
    CompilationConfig as CompilationConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.compilation import DynamicShapesType as DynamicShapesType
from vllm.config.utils import Range as Range, hash_factors as hash_factors
from vllm.logger import init_logger as init_logger
from vllm.logging_utils import lazy as lazy
from vllm.platforms import current_platform as current_platform
from vllm.tracing import (
    instrument as instrument,
    instrument_manual as instrument_manual,
)
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname

logger: Incomplete

def make_copy_and_call(
    sym_tensor_indices: list[int],
    input_buffers: list[torch.Tensor | None],
    callable_fn: Callable[..., Any],
) -> Callable[..., Any]: ...
def make_compiler(compilation_config: CompilationConfig) -> CompilerInterface: ...

class CompilerManager:
    cache: dict[tuple[Range, int, str], Any]
    is_cache_updated: bool
    compilation_config: Incomplete
    compiler: Incomplete
    loaded_artifacts: dict[str, Any]
    def __init__(self, compilation_config: CompilationConfig) -> None: ...
    def compute_hash(self, vllm_config: VllmConfig) -> str: ...
    @contextmanager
    def compile_context(self, compile_range: Range) -> Generator[None, None, None]: ...
    disable_cache: Incomplete
    cache_dir: Incomplete
    cache_file_path: Incomplete
    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None: ...
    def save_to_file(self) -> None: ...
    def load(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable[..., Any] | None: ...
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        additional_inductor_config: dict[str, Any],
        compilation_config: CompilationConfig,
        compile_range: Range,
        graph_index: int = 0,
        num_graphs: int = 1,
    ) -> Any: ...

class StopCompiling(BaseException): ...

@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule

def split_graph(
    graph: fx.GraphModule, splitting_ops: list[str]
) -> tuple[fx.GraphModule, list[SplitItem]]: ...

compilation_start_time: float

def wrap_with_cudagraph_if_needed(
    piecewise_backend: Any,
    vllm_config: VllmConfig,
    compilation_config: CompilationConfig,
    is_first_graph: bool,
    is_last_graph: bool,
) -> Any: ...

class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    compile_submod_names: Incomplete
    compilation_config: Incomplete
    vllm_config: Incomplete
    vllm_backend: Incomplete
    extra_traceback: bool
    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        vllm_config: VllmConfig,
        vllm_backend: VllmBackend,
    ) -> None: ...
    def run(self, *args: Any) -> Any: ...
    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any: ...

model_tag: str
model_is_encoder: bool

@contextmanager
def set_model_tag(
    tag: str, is_encoder: bool = False
) -> Generator[None, None, None]: ...

class VllmBackend:
    vllm_config: VllmConfig
    compilation_config: CompilationConfig
    graph: fx.GraphModule
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable[..., Any]
    post_grad_passes: Sequence[Callable[..., Any]]
    compiler_manager: CompilerManager
    inductor_config: dict[str, Any]
    prefix: Incomplete
    is_encoder: Incomplete
    pass_manager: Incomplete
    pass_key: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, prefix: str = "", is_encoder: bool = False
    ) -> None: ...
    def collect_standalone_compile_artifacts(
        self,
    ) -> tuple[Any, dict[str, list[int]] | None, dict[str, bool] | None]: ...
    def configure_post_pass(self) -> None: ...
    def __call__(self, graph: fx.GraphModule, example_inputs: Sequence[Any]) -> Any: ...
