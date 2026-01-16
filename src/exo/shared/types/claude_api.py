"""Claude Messages API types for request/response conversion."""

from typing import Literal

from pydantic import BaseModel, Field

# Type aliases
ClaudeRole = Literal["user", "assistant"]
ClaudeStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


# Content block types
class ClaudeTextBlock(BaseModel, frozen=True):
    """Text content block in Claude Messages API."""

    type: Literal["text"] = "text"
    text: str


class ClaudeImageSource(BaseModel, frozen=True):
    """Image source for Claude image blocks."""

    type: Literal["base64", "url"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class ClaudeImageBlock(BaseModel, frozen=True):
    """Image content block in Claude Messages API."""

    type: Literal["image"] = "image"
    source: ClaudeImageSource


ClaudeContentBlock = ClaudeTextBlock | ClaudeImageBlock


# Request types
class ClaudeMessage(BaseModel, frozen=True):
    """Message in Claude Messages API request."""

    role: ClaudeRole
    content: str | list[ClaudeContentBlock]


class ClaudeMessagesRequest(BaseModel):
    """Request body for Claude Messages API."""

    model: str
    max_tokens: int
    messages: list[ClaudeMessage]
    system: str | list[ClaudeTextBlock] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, str] | None = None


# Response types
class ClaudeUsage(BaseModel, frozen=True):
    """Token usage in Claude Messages API response."""

    input_tokens: int
    output_tokens: int


class ClaudeMessagesResponse(BaseModel, frozen=True):
    """Response body for Claude Messages API."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ClaudeTextBlock]
    model: str
    stop_reason: ClaudeStopReason | None = None
    stop_sequence: str | None = None
    usage: ClaudeUsage


# Streaming event types
class ClaudeMessageStart(BaseModel, frozen=True):
    """Partial message in message_start event."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ClaudeTextBlock] = Field(default_factory=list)
    model: str
    stop_reason: ClaudeStopReason | None = None
    stop_sequence: str | None = None
    usage: ClaudeUsage


class ClaudeMessageStartEvent(BaseModel, frozen=True):
    """Event sent at start of message stream."""

    type: Literal["message_start"] = "message_start"
    message: ClaudeMessageStart


class ClaudeContentBlockStartEvent(BaseModel, frozen=True):
    """Event sent at start of a content block."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ClaudeTextBlock


class ClaudeTextDelta(BaseModel, frozen=True):
    """Delta for text content block."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ClaudeContentBlockDeltaEvent(BaseModel, frozen=True):
    """Event sent for content block delta."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: ClaudeTextDelta


class ClaudeContentBlockStopEvent(BaseModel, frozen=True):
    """Event sent at end of a content block."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class ClaudeMessageDeltaUsage(BaseModel, frozen=True):
    """Usage in message_delta event."""

    output_tokens: int


class ClaudeMessageDelta(BaseModel, frozen=True):
    """Delta in message_delta event."""

    stop_reason: ClaudeStopReason | None = None
    stop_sequence: str | None = None


class ClaudeMessageDeltaEvent(BaseModel, frozen=True):
    """Event sent with final message delta."""

    type: Literal["message_delta"] = "message_delta"
    delta: ClaudeMessageDelta
    usage: ClaudeMessageDeltaUsage


class ClaudeMessageStopEvent(BaseModel, frozen=True):
    """Event sent at end of message stream."""

    type: Literal["message_stop"] = "message_stop"


ClaudeStreamEvent = (
    ClaudeMessageStartEvent
    | ClaudeContentBlockStartEvent
    | ClaudeContentBlockDeltaEvent
    | ClaudeContentBlockStopEvent
    | ClaudeMessageDeltaEvent
    | ClaudeMessageStopEvent
)
