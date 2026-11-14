from typing import Literal

from pydantic import Field

from exo.api.types import (
    CompletionTokensDetails,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.tensorfold import NativeModel


class LogprobPositions(NativeModel):
    """The raw token position of each answer logprob entry in the same chunk."""

    content: list[int] = Field(default_factory=list)


class StreamAnchor(NativeModel):
    version: Literal[1]
    token_count: int = Field(ge=0)
    token_id: int | None = Field(ge=0)
    logprob_positions: LogprobPositions | None = None


class TokenFrame(NativeModel):
    object: Literal["tensorfold.token.chunk"]
    version: Literal[1]
    id: str
    model: str
    offset: int = Field(ge=0)
    token_ids: list[int] = Field(min_length=1)


class FunctionDelta(NativeModel):
    name: str | None = None
    arguments: str | None = None


class ToolDelta(NativeModel):
    index: int = Field(ge=0, le=1023)
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionDelta = Field(default_factory=FunctionDelta)


class Delta(NativeModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolDelta] | None = None


class LogprobRow(NativeModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprobItem] = Field(default_factory=list)


class Logprobs(NativeModel):
    content: list[LogprobRow] = Field(default_factory=list)


class Choice(NativeModel):
    index: Literal[0] = 0
    delta: Delta = Field(default_factory=Delta)
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = (
        None
    )
    logprobs: Logprobs | None = None


class NativeUsage(NativeModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    prompt_tokens_details: PromptTokensDetails = Field(
        default_factory=PromptTokensDetails
    )
    completion_tokens_details: CompletionTokensDetails = Field(
        default_factory=CompletionTokensDetails
    )

    def as_exo(self) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            prompt_tokens_details=self.prompt_tokens_details,
            completion_tokens_details=self.completion_tokens_details,
        )


class RuntimeStats(NativeModel):
    token_ids: list[int]
    tokens_per_second: float = Field(ge=0)
    prefill_seconds: float | None = Field(default=None, ge=0)
    seconds: float = Field(default=0.0, ge=0)


class SemanticFrame(NativeModel):
    object: Literal["chat.completion.chunk"]
    id: str
    model: str
    choices: list[Choice]
    tensorfold_stream: StreamAnchor
    usage: NativeUsage | None = None
    tensorfold: RuntimeStats | None = None


class WireError(NativeModel):
    message: str


class ErrorFrame(NativeModel):
    error: WireError


class Envelope(NativeModel):
    object: str | None = None
    error: WireError | None = None
