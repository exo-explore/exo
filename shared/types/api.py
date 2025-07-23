from typing import Any, Literal

from pydantic import BaseModel

from shared.openai_compat import FinishReason


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class TopLogprobItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobsContentItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprobItem]


class Logprobs(BaseModel):
    content: list[LogprobsContentItem] | None = None


class StreamingChoiceResponse(BaseModel):
    index: int
    delta: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None



class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice | StreamingChoiceResponse]
    usage: Usage | None = None
    service_tier: str | None = None


class ChatCompletionTaskParams(BaseModel):
    model: str
    frequency_penalty: float | None = None
    messages: list[ChatCompletionMessage]
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None
