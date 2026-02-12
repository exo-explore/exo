"""Canonical internal type for text generation task parameters.

All external API formats (Chat Completions, Claude Messages, OpenAI Responses)
are converted to TextGenerationTaskParams at the API boundary via adapters.
"""

from typing import Any, Literal

from pydantic import BaseModel

from exo.shared.types.common import ModelId

MessageRole = Literal["user", "assistant", "system", "developer"]


class InputMessage(BaseModel, frozen=True):
    """Internal message for text generation pipelines."""

    role: MessageRole
    content: str


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
    enable_thinking: bool | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
