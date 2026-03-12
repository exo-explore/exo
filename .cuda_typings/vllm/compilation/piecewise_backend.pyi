import dataclasses
import torch.fx as fx
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from typing import Any
from vllm.compilation.backends import VllmBackend as VllmBackend
from vllm.config import VllmConfig as VllmConfig
from vllm.config.utils import Range as Range
from vllm.logger import init_logger as init_logger

logger: Incomplete

def get_fake_args_from_graph(graph: fx.GraphModule) -> list[Any]: ...
def create_concrete_args(graph: fx.GraphModule, size: int) -> list[Any]: ...
@dataclasses.dataclass
class RangeEntry:
    compile_range: Range
    compiled: bool = ...
    runnable: Callable[..., Any] = ...

class PiecewiseBackend:
    graph: Incomplete
    vllm_config: Incomplete
    compilation_config: Incomplete
    piecewise_compile_index: Incomplete
    total_piecewise_compiles: Incomplete
    vllm_backend: Incomplete
    compiled_runnables: Incomplete
    submod_name: Incomplete
    is_first_graph: Incomplete
    is_last_graph: Incomplete
    is_full_graph: Incomplete
    is_encoder_compilation: Incomplete
    compile_ranges: Incomplete
    compile_sizes: Incomplete
    sym_shape_indices: Incomplete
    returns_tuple: Incomplete
    range_entries: dict[Range, RangeEntry]
    def __init__(
        self,
        graph: fx.GraphModule | None,
        vllm_config: VllmConfig,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        vllm_backend: VllmBackend,
        returns_tuple: bool,
        compiled_runnables: dict[str, Callable[..., Any]] | None = None,
        submod_name: str = "",
    ) -> None: ...
    def get_compiled_graph_wrapper(
        self, compiled_graph: Callable[..., Any]
    ) -> Callable[..., Any]: ...
    def to_bytes(self) -> dict[str, bytes]: ...
    def compile_all_ranges(self) -> None: ...
    def load_all_ranges(self) -> None: ...
    def __call__(self, *args: Any) -> Any: ...
