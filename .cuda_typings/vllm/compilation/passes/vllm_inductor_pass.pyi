import torch
from .inductor_pass import InductorPass as InductorPass
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from dataclasses import dataclass
from torch._inductor.pattern_matcher import PatternMatcherPass as PatternMatcherPass
from typing import ClassVar
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger

logger: Incomplete

@dataclass
class InductorCompilationConfig:
    splitting_ops: list[str] | None = ...
    use_inductor_graph_partition: bool = ...

class VllmInductorPass(InductorPass):
    dump_prefix: ClassVar[int | None]
    compilation_config: Incomplete
    pass_config: Incomplete
    model_dtype: Incomplete
    device: str | None
    pass_name: Incomplete
    def __init__(self, config: VllmConfig) -> None: ...
    @staticmethod
    def time_and_log(
        call_fn: Callable[[VllmInductorPass, torch.fx.Graph], None],
    ) -> Callable[[VllmInductorPass, torch.fx.Graph], None]: ...
    def dump_graph(self, graph: torch.fx.Graph, stage: str) -> None: ...
    def begin(self) -> None: ...
    def end_and_log(self) -> None: ...

class VllmPatternMatcherPass(VllmInductorPass):
    matched_count: int
    def dump_patterns(
        self, config: VllmConfig, pm_pass: PatternMatcherPass
    ) -> None: ...

class PrinterInductorPass(VllmInductorPass):
    name: Incomplete
    def __init__(self, name: str, config: VllmConfig) -> None: ...
    def __call__(self, graph: torch.fx.Graph) -> None: ...
