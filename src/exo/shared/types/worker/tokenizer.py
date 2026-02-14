from collections.abc import Callable
from typing import Any, Protocol


class Tokenizer(Protocol):
    @property
    def has_tool_calling(self) -> bool: ...

    @property
    def has_call_start(self) -> bool: ...

    @property
    def tool_call_end(self) -> str | None: ...

    @property
    def tool_parser(self) -> Callable[[str], dict[str, Any] | list[dict[str, Any]]] | None: ...

    @property
    def think_start(self) -> str | None: ...

    @property
    def think_start_id(self) -> int | None: ...

    @property
    def think_end(self) -> int | None: ...


class MutableTokenizer(Tokenizer, Protocol):
    """Extended protocol for tokenizers whose tool-calling fields can be reassigned."""

    _tool_call_start: str | None
    _tool_call_end: str | None
    _tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]] | None
