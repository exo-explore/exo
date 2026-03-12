from pydantic import BaseModel
from typing import Any, Literal

class AnthropicError(BaseModel):
    type: str
    message: str

class AnthropicErrorResponse(BaseModel):
    type: Literal["error"]
    error: AnthropicError

class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None

class AnthropicContentBlock(BaseModel):
    type: Literal["text", "image", "tool_use", "tool_result", "thinking"]
    text: str | None
    source: dict[str, Any] | None
    id: str | None
    tool_use_id: str | None
    name: str | None
    input: dict[str, Any] | None
    content: str | list[dict[str, Any]] | None
    is_error: bool | None
    thinking: str | None
    signature: str | None

class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]

class AnthropicTool(BaseModel):
    name: str
    description: str | None
    input_schema: dict[str, Any]
    @classmethod
    def validate_input_schema(cls, v): ...

class AnthropicToolChoice(BaseModel):
    type: Literal["auto", "any", "tool", "none"]
    name: str | None
    def validate_name_required_for_tool(self) -> AnthropicToolChoice: ...

class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: dict[str, Any] | None
    stop_sequences: list[str] | None
    stream: bool | None
    system: str | list[AnthropicContentBlock] | None
    temperature: float | None
    tool_choice: AnthropicToolChoice | None
    tools: list[AnthropicTool] | None
    top_k: int | None
    top_p: float | None
    @classmethod
    def validate_model(cls, v): ...
    @classmethod
    def validate_max_tokens(cls, v): ...

class AnthropicDelta(BaseModel):
    type: (
        Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"]
        | None
    )
    text: str | None
    thinking: str | None
    partial_json: str | None
    signature: str | None
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    stop_sequence: str | None

class AnthropicStreamEvent(BaseModel):
    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: AnthropicMessagesResponse | None
    delta: AnthropicDelta | None
    content_block: AnthropicContentBlock | None
    index: int | None
    error: AnthropicError | None
    usage: AnthropicUsage | None

class AnthropicMessagesResponse(BaseModel):
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    stop_sequence: str | None
    usage: AnthropicUsage | None
    def model_post_init(self, /, __context) -> None: ...

class AnthropicContextManagement(BaseModel):
    original_input_tokens: int

class AnthropicCountTokensRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicContentBlock] | None
    tool_choice: AnthropicToolChoice | None
    tools: list[AnthropicTool] | None
    @classmethod
    def validate_model(cls, v): ...

class AnthropicCountTokensResponse(BaseModel):
    input_tokens: int
    context_management: AnthropicContextManagement | None
