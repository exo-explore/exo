from collections.abc import Callable
from typing import Any, final

_THINK_TOKENS: list[tuple[str, str]] = [
    ("<think>", "</think>"),
    ("<longcat_think>", "</longcat_think>"),
]


@final
class TokenizerWrapper:
    """Engine-agnostic tokenizer wrapper that satisfies the Tokenizer protocol.

    Wraps a raw HuggingFace tokenizer and adds tool-calling and thinking
    properties expected by the runner. All other attribute access is proxied
    to the underlying tokenizer.
    """

    _tokenizer: Any
    _tool_call_start: str | None
    _tool_call_end: str | None
    _tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]] | None
    _think_start: str | None
    _think_start_id: int | None
    _think_end_id: int | None

    def __init__(
        self,
        tokenizer: Any,  # pyright: ignore[reportAny]
        *,
        tool_call_start: str | None = None,
        tool_call_end: str | None = None,
        tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]] | None = None,
    ) -> None:
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_tool_call_start", tool_call_start)
        object.__setattr__(self, "_tool_call_end", tool_call_end)
        object.__setattr__(self, "_tool_parser", tool_parser)

        # Detect thinking tokens from vocabulary
        think_start: str | None = None
        think_start_id: int | None = None
        think_end_id: int | None = None
        vocab: dict[str, int] = tokenizer.get_vocab()  # pyright: ignore[reportAny]
        for start_tok, end_tok in _THINK_TOKENS:
            if start_tok in vocab and end_tok in vocab:
                think_start = start_tok
                think_start_id = vocab[start_tok]
                think_end_id = vocab[end_tok]
                break
        object.__setattr__(self, "_think_start", think_start)
        object.__setattr__(self, "_think_start_id", think_start_id)
        object.__setattr__(self, "_think_end_id", think_end_id)

        # Disable tool calling if tokens aren't in the vocabulary
        if (tool_call_start and tool_call_start not in vocab) or (
            tool_call_end and tool_call_end not in vocab
        ):
            object.__setattr__(self, "_tool_call_start", None)
            object.__setattr__(self, "_tool_call_end", None)
            object.__setattr__(self, "_tool_parser", None)

    @property
    def has_tool_calling(self) -> bool:
        return self._tool_call_start is not None

    @property
    def has_call_start(self) -> bool:
        return self._tool_call_start is not None

    @property
    def tool_call_start(self) -> str | None:
        return self._tool_call_start

    @property
    def tool_call_end(self) -> str | None:
        return self._tool_call_end

    @property
    def tool_parser(self) -> Callable[[str], dict[str, Any] | list[dict[str, Any]]] | None:
        return self._tool_parser

    @property
    def think_start(self) -> str | None:
        return self._think_start

    @property
    def think_start_id(self) -> int | None:
        return self._think_start_id

    @property
    def think_end(self) -> int | None:
        return self._think_end_id

    def __getattr__(self, name: str) -> Any:  # pyright: ignore[reportAny]
        return getattr(self._tokenizer, name)  # pyright: ignore[reportAny]

    def __setattr__(self, name: str, value: Any) -> None:  # pyright: ignore[reportAny]
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._tokenizer, name, value)  # pyright: ignore[reportAny]
