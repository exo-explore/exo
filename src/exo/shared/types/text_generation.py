"""Canonical internal type for text generation task parameters.

All external API formats (Chat Completions, Claude Messages, OpenAI Responses)
are converted to TextGenerationTaskParams at the API boundary via adapters.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, WrapValidator

from exo.shared.types.common import ModelId, TruncatingString

MessageRole = Literal["user", "assistant", "system", "developer", "tool"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


def resolve_reasoning_params(
    reasoning_effort: ReasoningEffort | None,
    enable_thinking: bool | None,
) -> tuple[ReasoningEffort | None, bool | None]:
    """
    enable_thinking=True  -> reasoning_effort="medium"
    enable_thinking=False -> reasoning_effort="none"
    reasoning_effort="none" -> enable_thinking=False
    reasoning_effort=<anything else> -> enable_thinking=True
    """
    resolved_effort: ReasoningEffort | None = reasoning_effort
    resolved_thinking: bool | None = enable_thinking

    if reasoning_effort is None and enable_thinking is not None:
        resolved_effort = "medium" if enable_thinking else "none"

    if enable_thinking is None and reasoning_effort is not None:
        resolved_thinking = reasoning_effort != "none"

    return resolved_effort, resolved_thinking


class InputMessageContent(TruncatingString):
    truncate_length = 100


class InputMessage(BaseModel, frozen=True):
    """Internal message for text generation pipelines."""

    role: MessageRole
    content: InputMessageContent


class Base64Image(TruncatingString):
    truncate_length = 10


class Base64ImageHash(TruncatingString):
    truncate_length = 10


def _wrap_chat_value(x: Any) -> Any:  # pyright: ignore[reportAny]
    if isinstance(x, (InputMessageContent, Base64Image)):
        return x
    if isinstance(x, str):
        return InputMessageContent(x)
    if isinstance(x, dict):
        return {k: _wrap_chat_value(v) for k, v in x.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(x, list):
        return [_wrap_chat_value(i) for i in x]  # pyright: ignore[reportUnknownVariableType]
    return x  # pyright: ignore[reportAny]


type ChatTemplateValue = Annotated[
    InputMessageContent
    | Base64Image
    | dict[str, ChatTemplateValue]
    | list[ChatTemplateValue]
    | str
    | int
    | float
    | MessageRole
    | bool,
    WrapValidator(lambda a, b: b(_wrap_chat_value(a))),  # pyright: ignore[reportAny]
]


class TextGenerationTaskParams(BaseModel, frozen=True):
    """Canonical internal task params for text generation.

    Every API adapter converts its wire type into this before handing
    off to the master/worker pipeline.
    """

    model: ModelId
    input: list[InputMessage]
    instructions: InputMessageContent | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    bench: bool = False
    use_prefix_cache: bool = False
    top_k: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    chat_template_messages: list[dict[str, ChatTemplateValue]] | None = None
    reasoning_effort: ReasoningEffort | None = None
    enable_thinking: bool | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    presence_penalty: float | None = None
    presence_context_size: int | None = None
    images: list[Base64Image] = Field(default_factory=list)
    image_hashes: dict[int, Base64ImageHash] = Field(default_factory=dict)
    total_input_chunks: int = 0
    image_count: int = 0
