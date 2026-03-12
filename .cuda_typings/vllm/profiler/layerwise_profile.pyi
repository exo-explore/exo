import types
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from dataclasses import dataclass, field
from torch._C._profiler import _ProfilerEvent
from torch.autograd.profiler import FunctionEvent as FunctionEvent
from torch.profiler import profile
from typing import Any, Generic, TypeAlias, TypeVar
from vllm.profiler.utils import (
    TablePrinter as TablePrinter,
    event_has_module as event_has_module,
    event_is_torch_op as event_is_torch_op,
    event_module_repr as event_module_repr,
    event_torch_op_stack_trace as event_torch_op_stack_trace,
    indent_string as indent_string,
)
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

@dataclass
class _ModuleTreeNode:
    event: _ProfilerEvent
    parent: _ModuleTreeNode | None = ...
    children: list["_ModuleTreeNode"] = field(default_factory=list)
    trace: str = ...
    @property
    def is_leaf(self): ...
    @property
    def is_torch_op(self): ...
    @property
    def is_cuda(self): ...

@dataclass
class SummaryStatsEntry:
    name: str
    cuda_time_us: float
    pct_cuda_time: float
    invocations: int

@dataclass
class ModelStatsEntry:
    name: str
    cpu_time_us: float
    cuda_time_us: float
    pct_cuda_time: float
    trace: str

StatsEntry: TypeAlias = ModelStatsEntry | SummaryStatsEntry
StatsEntryT = TypeVar("StatsEntryT", bound=StatsEntry)

@dataclass
class _StatsTreeNode(Generic[StatsEntryT]):
    entry: StatsEntryT
    children: list["_StatsTreeNode[StatsEntryT]"] = field(default_factory=list)
    parent: _StatsTreeNode[StatsEntryT] | None = ...

@dataclass
class LayerwiseProfileResults(profile):
    num_running_seqs: int | None = ...
    def __post_init__(self) -> None: ...
    def print_model_table(self, column_widths: dict[str, int] | None = None): ...
    def print_summary_table(self, column_widths: dict[str, int] | None = None): ...
    def export_model_stats_table_csv(self, filename: str): ...
    def export_summary_stats_table_csv(self, filename: str): ...
    def convert_stats_to_dict(self) -> dict[str, Any]: ...

class layerwise_profile(profile):
    num_running_seqs: Incomplete
    def __init__(self, num_running_seqs: int | None = None) -> None: ...
    def __enter__(self): ...
    results: Incomplete
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...
