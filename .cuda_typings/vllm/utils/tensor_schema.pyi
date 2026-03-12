from _typeshed import Incomplete
from typing import Any
from vllm.logger import init_logger as init_logger

logger: Incomplete

class TensorShape:
    dims: Incomplete
    dynamic_dims: Incomplete
    def __init__(
        self, *dims: int | str, dynamic_dims: set[str] | None = None
    ) -> None: ...
    def resolve(self, **bindings: int) -> tuple[int | str, ...]: ...

class TensorSchema:
    def __init__(
        self,
        *,
        validate: bool = True,
        resolve_bindings: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def validate(self) -> None: ...
    def print_shapes(self) -> None: ...
