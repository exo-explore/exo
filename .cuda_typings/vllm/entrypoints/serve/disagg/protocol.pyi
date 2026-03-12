from pydantic import BaseModel
from typing import Any
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProbs as ChatCompletionLogProbs,
)
from vllm.entrypoints.openai.engine.protocol import (
    SamplingParams as SamplingParams,
    StreamOptions as StreamOptions,
)
from vllm.logprobs import Logprob as Logprob
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.utils import random_uuid as random_uuid

class GenerateRequest(BaseModel):
    request_id: str
    token_ids: list[int]
    features: str | None
    sampling_params: SamplingParams
    model: str | None
    stream: bool | None
    stream_options: StreamOptions | None
    cache_salt: str | None
    priority: int
    kv_transfer_params: dict[str, Any] | None
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...

class GenerateResponseChoice(BaseModel):
    index: int
    logprobs: ChatCompletionLogProbs | None
    finish_reason: str | None
    token_ids: list[int] | None

class GenerateResponse(BaseModel):
    request_id: str
    choices: list[GenerateResponseChoice]
    prompt_logprobs: list[dict[int, Logprob] | None] | None
    kv_transfer_params: dict[str, Any] | None
