import contextlib
import torch
from _typeshed import Incomplete
from collections.abc import Generator
from vllm.logger import init_logger as init_logger

logger: Incomplete

def should_split(node: torch.fx.Node, splitting_ops: list[str]) -> bool: ...
@contextlib.contextmanager
def inductor_partition_rule_context(
    splitting_ops: list[str] | None,
) -> Generator[None, None, None]: ...
