import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Generator
from contextlib import contextmanager
from torch import fx as fx
from torch._inductor.custom_graph_pass import CustomGraphPass
from typing import Any, ParamSpec, TypeVar
from vllm.config.utils import Range as Range

P = ParamSpec("P")
R = TypeVar("R")

class PassContext:
    compile_range: Range
    def __init__(self, compile_range: Range) -> None: ...

def get_pass_context() -> PassContext: ...
@contextmanager
def pass_context(compile_range: Range) -> Generator[None, None, None]: ...

class InductorPass(CustomGraphPass):
    def uuid(self) -> str: ...
    @staticmethod
    def hash_source(*srcs: str | Any) -> str: ...
    @staticmethod
    def hash_dict(dict_: dict[Any, Any]) -> str: ...
    def is_applicable_for_range(self, compile_range: Range) -> bool: ...

class CallableInductorPass(InductorPass):
    callable: Incomplete
    def __init__(
        self, callable: Callable[[fx.Graph], None], uuid: Any | None = None
    ) -> None: ...
    def __call__(self, graph: torch.fx.Graph) -> None: ...
    def uuid(self) -> Any: ...

def enable_fake_mode(fn: Callable[P, R]) -> Callable[P, R]: ...
