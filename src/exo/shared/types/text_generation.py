"""Canonical internal type for text generation task parameters.

All external API formats (Chat Completions, Claude Messages, OpenAI Responses)
are converted to TextGenerationTaskParams at the API boundary via adapters.
"""

from typing import Any, Literal, cast

from pydantic import BaseModel, Field, field_validator

from exo.shared.types.common import ModelId

MessageRole = Literal["user", "assistant", "system", "developer"]
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


class InputMessageContent(str):
    def __repr__(self):
        return f"<InputMessageContent {self[:100]}...>"


class InputMessage(BaseModel, frozen=True):
    """Internal message for text generation pipelines."""

    role: MessageRole
    content: str

    @field_validator("content", mode="after")
    @classmethod
    def _wrap_content(cls, v: str) -> str:
        return InputMessageContent(v)


class Base64Image(str):
    def __repr__(self):
        return f"<Base64Image: {self[:10]}...>"


class TextGenerationTaskParams(BaseModel, frozen=True):
    """Canonical internal task params for text generation.

    Every API adapter converts its wire type into this before handing
    off to the master/worker pipeline.
    """

    model: ModelId
    input: list[InputMessage]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    bench: bool = False
    top_k: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    chat_template_messages: list[dict[str, Any]] | None = None
    reasoning_effort: ReasoningEffort | None = None
    enable_thinking: bool | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    images: list[str] = Field(default_factory=list)
    image_hashes: dict[int, str] = Field(default_factory=dict)
    total_input_chunks: int = 0
    image_count: int = 0

    @field_validator("images", mode="after")
    @classmethod
    def _wrap_images(cls, v: list[str]) -> list[str]:
        return [Base64Image(x) for x in v]

    @field_validator("image_hashes", mode="after")
    @classmethod
    def _wrap_image_hashes(cls, v: dict[int, str]) -> dict[int, str]:
        return {k: Base64Image(x) for k, x in v.items()}

    @field_validator("instructions", mode="after")
    @classmethod
    def _wrap_instructions(cls, v: str | None) -> str | None:
        return InputMessageContent(v) if v is not None else None

    @field_validator("chat_template_messages", mode="after")
    @classmethod
    def _wrap_chat_template_messages(
        cls, v: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        if v is None:
            return None

        def wrap(x: object) -> object:
            if isinstance(x, (InputMessageContent, Base64Image)):
                return x
            if isinstance(x, str):
                return InputMessageContent(x)
            if isinstance(x, dict):
                return {
                    k: wrap(val) for k, val in cast(dict[object, object], x).items()
                }
            if isinstance(x, list):
                return [wrap(i) for i in cast(list[object], x)]
            return x

        return cast(list[dict[str, Any]], wrap(v))
