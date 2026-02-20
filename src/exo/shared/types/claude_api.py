"""Claude Messages API types for request/response conversion."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from exo.shared.types.common import ModelId

# Tool definition types
ClaudeToolInputSchema = dict[str, Any]


class ClaudeToolDefinition(BaseModel, frozen=True):
    """Tool definition in Claude Messages API request."""

    name: str
    description: str | None = None
    input_schema: ClaudeToolInputSchema


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


class ClaudeThinkingBlock(BaseModel, frozen=True):
    """Thinking content block in Claude Messages API."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ClaudeToolUseBlock(BaseModel, frozen=True):
    """Tool use content block in Claude Messages API."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ClaudeToolResultBlock(BaseModel, frozen=True):
    """Tool result content block in Claude Messages API request."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[ClaudeTextBlock] | None = None
    is_error: bool | None = None
    cache_control: dict[str, str] | None = None


ClaudeContentBlock = (
    ClaudeTextBlock | ClaudeImageBlock | ClaudeThinkingBlock | ClaudeToolUseBlock
)

# Input content blocks can also include tool_result (sent by user after tool_use)
ClaudeInputContentBlock = (
    ClaudeTextBlock
    | ClaudeImageBlock
    | ClaudeThinkingBlock
    | ClaudeToolUseBlock
    | ClaudeToolResultBlock
)


# Request types
class ClaudeMessage(BaseModel, frozen=True):
    """Message in Claude Messages API request."""

    role: ClaudeRole
    content: str | list[ClaudeInputContentBlock]


class ClaudeThinkingConfig(BaseModel, frozen=True):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: int | None = None


class ClaudeMessagesRequest(BaseModel):
    """Request body for Claude Messages API."""

    model: ModelId
    max_tokens: int
    messages: list[ClaudeMessage]
    system: str | list[ClaudeTextBlock] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[ClaudeToolDefinition] | None = None
    metadata: dict[str, str] | None = None
    thinking: ClaudeThinkingConfig | None = None


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
    content: list[ClaudeContentBlock]
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
    content_block: ClaudeTextBlock | ClaudeThinkingBlock | ClaudeToolUseBlock


class ClaudeTextDelta(BaseModel, frozen=True):
    """Delta for text content block."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ClaudeThinkingDelta(BaseModel, frozen=True):
    """Delta for thinking content block."""

    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class ClaudeInputJsonDelta(BaseModel, frozen=True):
    """Delta for tool use input JSON content block."""

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ClaudeContentBlockDeltaEvent(BaseModel, frozen=True):
    """Event sent for content block delta."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: ClaudeTextDelta | ClaudeThinkingDelta | ClaudeInputJsonDelta


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
