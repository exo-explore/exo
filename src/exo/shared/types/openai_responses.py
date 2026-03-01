"""OpenAI Responses API wire types.

These types model the OpenAI Responses API request/response format.
ResponsesRequest is the API-level wire type; for the canonical internal
task params type used by the inference pipeline, see
``exo.shared.types.text_generation.TextGenerationTaskParams``.
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from exo.shared.types.common import ModelId

# Type aliases
ResponseStatus = Literal["completed", "failed", "in_progress", "incomplete"]
ResponseRole = Literal["user", "assistant", "system", "developer"]


# Request input content part types
class ResponseInputTextPart(BaseModel, frozen=True):
    """Text content part in a Responses API input message."""

    type: Literal["input_text"] = "input_text"
    text: str


class ResponseOutputTextPart(BaseModel, frozen=True):
    """Output text content part (used when replaying assistant messages in input)."""

    type: Literal["output_text"] = "output_text"
    text: str


ResponseContentPart = ResponseInputTextPart | ResponseOutputTextPart


# Request input item types
class ResponseInputMessage(BaseModel, frozen=True):
    """Input message for Responses API."""

    role: ResponseRole
    content: str | list[ResponseContentPart]
    type: Literal["message"] = "message"


class FunctionCallInputItem(BaseModel, frozen=True):
    """Function call item replayed in input (from a previous assistant response)."""

    type: Literal["function_call"] = "function_call"
    id: str | None = None
    call_id: str
    name: str
    arguments: str
    status: ResponseStatus | None = None


class FunctionCallOutputInputItem(BaseModel, frozen=True):
    """Function call output item in input (user providing tool results)."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str
    id: str | None = None
    status: ResponseStatus | None = None


ResponseInputItem = (
    ResponseInputMessage | FunctionCallInputItem | FunctionCallOutputInputItem
)


class ResponsesRequest(BaseModel, frozen=True):
    """Request body for OpenAI Responses API.

    This is the API wire type for the Responses endpoint. The canonical
    internal task params type is ``TextGenerationTaskParams``; see the
    ``responses_request_to_text_generation`` adapter for conversion.
    """

    # --- OpenAI Responses API standard fields ---
    model: ModelId
    input: str | list[ResponseInputItem]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, str] | None = None

    # --- exo extensions (not in OpenAI Responses API spec) ---
    top_k: int | None = Field(
        default=None,
        description="[exo extension] Top-k sampling parameter. Not part of the OpenAI Responses API.",
        json_schema_extra={"x-exo-extension": True},
    )
    stop: str | list[str] | None = Field(
        default=None,
        description="[exo extension] Stop sequence(s). Not part of the OpenAI Responses API.",
        json_schema_extra={"x-exo-extension": True},
    )
    seed: int | None = Field(
        default=None,
        description="[exo extension] Seed for deterministic sampling. Not part of the OpenAI Responses API.",
        json_schema_extra={"x-exo-extension": True},
    )

    # --- Internal fields (preserved during serialization, hidden from OpenAPI schema) ---
    chat_template_messages: list[dict[str, Any]] | None = Field(
        default=None,
        description="Internal: pre-formatted messages for tokenizer chat template. Not part of the OpenAI Responses API.",
        json_schema_extra={"x-exo-internal": True},
    )


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


class ResponseFunctionCallItem(BaseModel, frozen=True):
    """Function call item in response output array."""

    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str
    status: ResponseStatus = "completed"


class ResponseReasoningSummaryText(BaseModel, frozen=True):
    """Summary text part in a reasoning output item."""

    type: Literal["summary_text"] = "summary_text"
    text: str


class ResponseReasoningItem(BaseModel, frozen=True):
    """Reasoning output item in response output array."""

    type: Literal["reasoning"] = "reasoning"
    id: str
    summary: list[ResponseReasoningSummaryText] = Field(default_factory=list)
    status: ResponseStatus = "completed"


ResponseItem = ResponseMessageItem | ResponseFunctionCallItem | ResponseReasoningItem


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
    sequence_number: int
    response: ResponsesResponse


class ResponseInProgressEvent(BaseModel, frozen=True):
    """Event sent when response starts processing."""

    type: Literal["response.in_progress"] = "response.in_progress"
    sequence_number: int
    response: ResponsesResponse


class ResponseOutputItemAddedEvent(BaseModel, frozen=True):
    """Event sent when an output item is added."""

    type: Literal["response.output_item.added"] = "response.output_item.added"
    sequence_number: int
    output_index: int
    item: ResponseItem


class ResponseContentPartAddedEvent(BaseModel, frozen=True):
    """Event sent when a content part is added."""

    type: Literal["response.content_part.added"] = "response.content_part.added"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    part: ResponseOutputText


class ResponseTextDeltaEvent(BaseModel, frozen=True):
    """Event sent for text delta during streaming."""

    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseTextDoneEvent(BaseModel, frozen=True):
    """Event sent when text content is done."""

    type: Literal["response.output_text.done"] = "response.output_text.done"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseContentPartDoneEvent(BaseModel, frozen=True):
    """Event sent when a content part is done."""

    type: Literal["response.content_part.done"] = "response.content_part.done"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    part: ResponseOutputText


class ResponseOutputItemDoneEvent(BaseModel, frozen=True):
    """Event sent when an output item is done."""

    type: Literal["response.output_item.done"] = "response.output_item.done"
    sequence_number: int
    output_index: int
    item: ResponseItem


class ResponseFunctionCallArgumentsDeltaEvent(BaseModel, frozen=True):
    """Event sent for function call arguments delta during streaming."""

    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    sequence_number: int
    item_id: str
    output_index: int
    delta: str


class ResponseFunctionCallArgumentsDoneEvent(BaseModel, frozen=True):
    """Event sent when function call arguments are complete."""

    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    sequence_number: int
    item_id: str
    output_index: int
    name: str
    arguments: str


class ResponseReasoningSummaryPartAddedEvent(BaseModel, frozen=True):
    """Event sent when a reasoning summary part is added."""

    type: Literal["response.reasoning_summary_part.added"] = (
        "response.reasoning_summary_part.added"
    )
    sequence_number: int
    item_id: str
    output_index: int
    summary_index: int
    part: ResponseReasoningSummaryText


class ResponseReasoningSummaryTextDeltaEvent(BaseModel, frozen=True):
    """Event sent for reasoning summary text delta during streaming."""

    type: Literal["response.reasoning_summary_text.delta"] = (
        "response.reasoning_summary_text.delta"
    )
    sequence_number: int
    item_id: str
    output_index: int
    summary_index: int
    delta: str


class ResponseReasoningSummaryTextDoneEvent(BaseModel, frozen=True):
    """Event sent when reasoning summary text is done."""

    type: Literal["response.reasoning_summary_text.done"] = (
        "response.reasoning_summary_text.done"
    )
    sequence_number: int
    item_id: str
    output_index: int
    summary_index: int
    text: str


class ResponseReasoningSummaryPartDoneEvent(BaseModel, frozen=True):
    """Event sent when a reasoning summary part is done."""

    type: Literal["response.reasoning_summary_part.done"] = (
        "response.reasoning_summary_part.done"
    )
    sequence_number: int
    item_id: str
    output_index: int
    summary_index: int
    part: ResponseReasoningSummaryText


class ResponseCompletedEvent(BaseModel, frozen=True):
    """Event sent when response is completed."""

    type: Literal["response.completed"] = "response.completed"
    sequence_number: int
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
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ResponseReasoningSummaryPartAddedEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseReasoningSummaryPartDoneEvent
    | ResponseCompletedEvent
)
