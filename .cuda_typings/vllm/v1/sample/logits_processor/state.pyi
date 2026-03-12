from collections.abc import Iterable, Iterator
from vllm.v1.sample.logits_processor.interface import (
    AddedRequest as AddedRequest,
    BatchUpdate as BatchUpdate,
    LogitsProcessor as LogitsProcessor,
    MovedRequest as MovedRequest,
    RemovedRequest as RemovedRequest,
)

class BatchUpdateBuilder:
    added: list[AddedRequest]
    moved: list[MovedRequest]
    batch_changed: bool
    def __init__(
        self,
        removed: list[RemovedRequest] | None = None,
        added: list[AddedRequest] | None = None,
        moved: list[MovedRequest] | None = None,
    ) -> None: ...
    @property
    def removed(self) -> list[RemovedRequest]: ...
    def removed_append(self, index: int) -> None: ...
    def has_removed(self) -> bool: ...
    def peek_removed(self) -> int | None: ...
    def pop_removed(self) -> int | None: ...
    def reset(self) -> bool: ...
    def get_and_reset(self, batch_size: int) -> BatchUpdate | None: ...

class LogitsProcessors:
    argmax_invariant: list[LogitsProcessor]
    non_argmax_invariant: list[LogitsProcessor]
    def __init__(
        self, logitsprocs: Iterable["LogitsProcessor"] | None = None
    ) -> None: ...
    @property
    def all(self) -> Iterator["LogitsProcessor"]: ...
