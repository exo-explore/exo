from _typeshed import Incomplete
from collections.abc import Generator, Hashable
from contextlib import contextmanager
from logging import Logger
from typing import Any, Literal
from vllm.logging_utils import (
    ColoredFormatter as ColoredFormatter,
    NewLineFormatter as NewLineFormatter,
)

DEFAULT_LOGGING_CONFIG: dict[str, dict[str, Any] | Any]
LogScope: Incomplete

class _VllmLogger(Logger):
    def debug_once(
        self, msg: str, *args: Hashable, scope: LogScope = "process"
    ) -> None: ...
    def info_once(
        self, msg: str, *args: Hashable, scope: LogScope = "process"
    ) -> None: ...
    def warning_once(
        self, msg: str, *args: Hashable, scope: LogScope = "process"
    ) -> None: ...

def init_logger(name: str) -> _VllmLogger: ...
@contextmanager
def suppress_logging(level: int = ...) -> Generator[None, Any, None]: ...
def current_formatter_type(logger: Logger) -> Literal["color", "newline", None]: ...

logger: Incomplete

def enable_trace_function_call(log_file_path: str, root_dir: str | None = None): ...
