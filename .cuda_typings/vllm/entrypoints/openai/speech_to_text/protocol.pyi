from _typeshed import Incomplete
from fastapi import UploadFile as UploadFile
from typing import Literal, TypeAlias
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage as DeltaMessage,
    OpenAIBaseModel as OpenAIBaseModel,
    UsageInfo as UsageInfo,
)
from vllm.exceptions import VLLMValidationError as VLLMValidationError
from vllm.logger import init_logger as init_logger
from vllm.sampling_params import (
    BeamSearchParams as BeamSearchParams,
    RequestOutputKind as RequestOutputKind,
    SamplingParams as SamplingParams,
)
from vllm.utils import random_uuid as random_uuid

logger: Incomplete

class TranscriptionResponseStreamChoice(OpenAIBaseModel):
    delta: DeltaMessage
    finish_reason: str | None
    stop_reason: int | str | None

class TranscriptionStreamResponse(OpenAIBaseModel):
    id: str
    object: Literal["transcription.chunk"]
    created: int
    model: str
    choices: list[TranscriptionResponseStreamChoice]
    usage: UsageInfo | None

AudioResponseFormat: TypeAlias

class TranscriptionRequest(OpenAIBaseModel):
    file: UploadFile
    model: str | None
    language: str | None
    prompt: str
    response_format: AudioResponseFormat
    timestamp_granularities: list[Literal["word", "segment"]]
    stream: bool | None
    stream_include_usage: bool | None
    stream_continuous_usage_stats: bool | None
    vllm_xargs: dict[str, str | int | float] | None
    to_language: str | None
    use_beam_search: bool
    n: int
    length_penalty: float
    include_stop_str_in_output: bool
    temperature: float
    top_p: float | None
    top_k: int | None
    min_p: float | None
    seed: int | None
    frequency_penalty: float | None
    repetition_penalty: float | None
    presence_penalty: float | None
    max_completion_tokens: int | None
    def to_beam_search_params(
        self, default_max_tokens: int, default_sampling_params: dict | None = None
    ) -> BeamSearchParams: ...
    def to_sampling_params(
        self, default_max_tokens: int, default_sampling_params: dict | None = None
    ) -> SamplingParams: ...
    @classmethod
    def validate_transcription_request(cls, data): ...

class TranscriptionUsageAudio(OpenAIBaseModel):
    type: Literal["duration"]
    seconds: int

class TranscriptionResponse(OpenAIBaseModel):
    text: str
    usage: TranscriptionUsageAudio

class TranscriptionWord(OpenAIBaseModel):
    end: float
    start: float
    word: str

class TranscriptionSegment(OpenAIBaseModel):
    id: int
    avg_logprob: float
    compression_ratio: float
    end: float
    no_speech_prob: float | None
    seek: int
    start: float
    temperature: float
    text: str
    tokens: list[int]

class TranscriptionResponseVerbose(OpenAIBaseModel):
    duration: str
    language: str
    text: str
    segments: list[TranscriptionSegment] | None
    words: list[TranscriptionWord] | None

TranscriptionResponseVariant: TypeAlias = (
    TranscriptionResponse | TranscriptionResponseVerbose
)

class TranslationResponseStreamChoice(OpenAIBaseModel):
    delta: DeltaMessage
    finish_reason: str | None
    stop_reason: int | str | None

class TranslationStreamResponse(OpenAIBaseModel):
    id: str
    object: Literal["translation.chunk"]
    created: int
    model: str
    choices: list[TranslationResponseStreamChoice]
    usage: UsageInfo | None

class TranslationRequest(OpenAIBaseModel):
    file: UploadFile
    model: str | None
    prompt: str
    response_format: AudioResponseFormat
    use_beam_search: bool
    n: int
    length_penalty: float
    include_stop_str_in_output: bool
    seed: int | None
    temperature: float
    language: str | None
    to_language: str | None
    stream: bool | None
    stream_include_usage: bool | None
    stream_continuous_usage_stats: bool | None
    max_completion_tokens: int | None
    def to_beam_search_params(
        self, default_max_tokens: int, default_sampling_params: dict | None = None
    ) -> BeamSearchParams: ...
    def to_sampling_params(
        self, default_max_tokens: int, default_sampling_params: dict | None = None
    ) -> SamplingParams: ...
    @classmethod
    def validate_stream_options(cls, data): ...

class TranslationResponse(OpenAIBaseModel):
    text: str

class TranslationWord(OpenAIBaseModel):
    end: float
    start: float
    word: str

class TranslationSegment(OpenAIBaseModel):
    id: int
    avg_logprob: float
    compression_ratio: float
    end: float
    no_speech_prob: float | None
    seek: int
    start: float
    temperature: float
    text: str
    tokens: list[int]

class TranslationResponseVerbose(OpenAIBaseModel):
    duration: str
    language: str
    text: str
    segments: list[TranslationSegment] | None
    words: list[TranslationWord] | None

TranslationResponseVariant: TypeAlias = TranslationResponse | TranslationResponseVerbose
