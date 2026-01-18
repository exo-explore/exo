"""OpenAI Responses API types for request/response conversion.

ResponsesRequest serves as both:
1. The external API request type for /v1/responses
2. The canonical internal type used throughout the inference pipeline

All external API formats (Chat Completions, Claude) are converted to
ResponsesRequest at the API boundary.
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

# Type aliases
ResponseStatus = Literal["completed", "failed", "in_progress", "incomplete"]
ResponseRole = Literal["user", "assistant", "system", "developer"]


# Request types
class ResponseInputMessage(BaseModel, frozen=True):
    """Input message for Responses API.

    This is also used as the internal message format throughout the pipeline.
    """

    role: ResponseRole
    content: str


class ResponsesRequest(BaseModel):
    """Request body for OpenAI Responses API.

    This is also the canonical internal task params format used throughout
    the inference pipeline. All external API formats are converted to this
    format at the API boundary.

    Field mapping from other APIs:
    - input: Replaces 'messages' from Chat Completions
    - instructions: System message, extracted from messages or Claude's 'system'
    - max_output_tokens: Replaces 'max_tokens' from Chat Completions
    """

    model: str
    input: str | list[ResponseInputMessage]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    # Tools support
    tools: list[dict[str, Any]] | None = None
    # previous_response_id not supported in MVP
    metadata: dict[str, str] | None = None


# Response types
class ResponseOutputText(BaseModel, frozen=True):
    """Text content in response output."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, str]] = Field(default_factory=list)


class ResponseMessageItem(BaseModel, frozen=True):
    """Message item in response output array."""

    type: Literal["message"] = "message"
    id: str
    role: Literal["assistant"] = "assistant"
    content: list[ResponseOutputText]
    status: ResponseStatus = "completed"


ResponseItem = ResponseMessageItem  # Can expand for function_call, reasoning, etc.


class ResponseUsage(BaseModel, frozen=True):
    """Token usage in Responses API response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponsesResponse(BaseModel, frozen=True):
    """Response body for OpenAI Responses API."""

    id: str
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: ResponseStatus = "completed"
    model: str
    output: list[ResponseItem]
    output_text: str
    usage: ResponseUsage | None = None


# Streaming event types
class ResponseCreatedEvent(BaseModel, frozen=True):
    """Event sent when response is created."""

    type: Literal["response.created"] = "response.created"
    response: ResponsesResponse


class ResponseInProgressEvent(BaseModel, frozen=True):
    """Event sent when response starts processing."""

    type: Literal["response.in_progress"] = "response.in_progress"
    response: ResponsesResponse


class ResponseOutputItemAddedEvent(BaseModel, frozen=True):
    """Event sent when an output item is added."""

    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int
    item: ResponseItem


class ResponseContentPartAddedEvent(BaseModel, frozen=True):
    """Event sent when a content part is added."""

    type: Literal["response.content_part.added"] = "response.content_part.added"
    output_index: int
    content_index: int
    part: ResponseOutputText


class ResponseTextDeltaEvent(BaseModel, frozen=True):
    """Event sent for text delta during streaming."""

    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    output_index: int
    content_index: int
    delta: str


class ResponseTextDoneEvent(BaseModel, frozen=True):
    """Event sent when text content is done."""

    type: Literal["response.output_text.done"] = "response.output_text.done"
    output_index: int
    content_index: int
    text: str


class ResponseContentPartDoneEvent(BaseModel, frozen=True):
    """Event sent when a content part is done."""

    type: Literal["response.content_part.done"] = "response.content_part.done"
    output_index: int
    content_index: int
    part: ResponseOutputText


class ResponseOutputItemDoneEvent(BaseModel, frozen=True):
    """Event sent when an output item is done."""

    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int
    item: ResponseItem


class ResponseCompletedEvent(BaseModel, frozen=True):
    """Event sent when response is completed."""

    type: Literal["response.completed"] = "response.completed"
    response: ResponsesResponse


ResponsesStreamEvent = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseOutputItemAddedEvent
    | ResponseContentPartAddedEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseContentPartDoneEvent
    | ResponseOutputItemDoneEvent
    | ResponseCompletedEvent
)
