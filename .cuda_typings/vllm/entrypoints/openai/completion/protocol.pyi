from _typeshed import Incomplete
from typing import Annotated, Any, Literal
from vllm.config import ModelConfig as ModelConfig
from vllm.config.utils import replace as replace
from vllm.entrypoints.openai.engine.protocol import (
    AnyResponseFormat as AnyResponseFormat,
    LegacyStructuralTagResponseFormat as LegacyStructuralTagResponseFormat,
    OpenAIBaseModel as OpenAIBaseModel,
    StreamOptions as StreamOptions,
    StructuralTagResponseFormat as StructuralTagResponseFormat,
    UsageInfo as UsageInfo,
)
from vllm.exceptions import VLLMValidationError as VLLMValidationError
from vllm.logger import init_logger as init_logger
from vllm.logprobs import Logprob as Logprob
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.sampling_params import (
    BeamSearchParams as BeamSearchParams,
    RepetitionDetectionParams as RepetitionDetectionParams,
    RequestOutputKind as RequestOutputKind,
    SamplingParams as SamplingParams,
    StructuredOutputsParams as StructuredOutputsParams,
)
from vllm.utils import random_uuid as random_uuid

logger: Incomplete

class CompletionRequest(OpenAIBaseModel):
    model: str | None
    prompt: (
        list[Annotated[int, None]]
        | list[list[Annotated[int, None]]]
        | str
        | list[str]
        | None
    )
    echo: bool | None
    frequency_penalty: float | None
    logit_bias: dict[str, float] | None
    logprobs: int | None
    max_tokens: int | None
    n: int
    presence_penalty: float | None
    seed: int | None
    stop: str | list[str] | None
    stream: bool | None
    stream_options: StreamOptions | None
    suffix: str | None
    temperature: float | None
    top_p: float | None
    user: str | None
    use_beam_search: bool
    top_k: int | None
    min_p: float | None
    repetition_penalty: float | None
    length_penalty: float
    stop_token_ids: list[int] | None
    include_stop_str_in_output: bool
    ignore_eos: bool
    min_tokens: int
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    truncate_prompt_tokens: Annotated[int, None] | None
    allowed_token_ids: list[int] | None
    prompt_logprobs: int | None
    prompt_embeds: bytes | list[bytes] | None
    add_special_tokens: bool
    response_format: AnyResponseFormat | None
    structured_outputs: StructuredOutputsParams | None
    priority: int
    request_id: str
    return_tokens_as_token_ids: bool | None
    return_token_ids: bool | None
    cache_salt: str | None
    kv_transfer_params: dict[str, Any] | None
    vllm_xargs: dict[str, str | int | float] | None
    repetition_detection: RepetitionDetectionParams | None
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...
    def to_beam_search_params(
        self, max_tokens: int, default_sampling_params: dict | None = None
    ) -> BeamSearchParams: ...
    def to_sampling_params(
        self, max_tokens: int, default_sampling_params: dict | None = None
    ) -> SamplingParams: ...
    @classmethod
    def validate_response_format(cls, data): ...
    @classmethod
    def check_structured_outputs_count(cls, data): ...
    @classmethod
    def check_logprobs(cls, data): ...
    @classmethod
    def validate_stream_options(cls, data): ...
    @classmethod
    def validate_prompt_and_prompt_embeds(cls, data): ...
    @classmethod
    def check_cache_salt_support(cls, data): ...

class CompletionLogProbs(OpenAIBaseModel):
    text_offset: list[int]
    token_logprobs: list[float | None]
    tokens: list[str]
    top_logprobs: list[dict[str, float] | None]

class CompletionResponseChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: CompletionLogProbs | None
    finish_reason: str | None
    stop_reason: int | str | None
    token_ids: list[int] | None
    prompt_logprobs: list[dict[int, Logprob] | None] | None
    prompt_token_ids: list[int] | None

class CompletionResponse(OpenAIBaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[CompletionResponseChoice]
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None
    system_fingerprint: str | None
    usage: UsageInfo
    kv_transfer_params: dict[str, Any] | None

class CompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: CompletionLogProbs | None
    finish_reason: str | None
    stop_reason: int | str | None
    prompt_token_ids: list[int] | None
    token_ids: list[int] | None

class CompletionStreamResponse(OpenAIBaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[CompletionResponseStreamChoice]
    usage: UsageInfo | None
