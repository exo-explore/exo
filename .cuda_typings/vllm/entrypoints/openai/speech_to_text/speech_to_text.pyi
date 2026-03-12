from _typeshed import Incomplete
from collections.abc import Callable as Callable
from fastapi import Request as Request
from functools import cached_property as cached_property
from typing import Final, Literal, TypeAlias, TypeVar
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage as DeltaMessage,
    ErrorResponse as ErrorResponse,
    RequestResponseMetadata as RequestResponseMetadata,
    UsageInfo as UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    OpenAIServing as OpenAIServing,
    SpeechToTextRequest as SpeechToTextRequest,
)
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionResponse as TranscriptionResponse,
    TranscriptionResponseStreamChoice as TranscriptionResponseStreamChoice,
    TranscriptionResponseVerbose as TranscriptionResponseVerbose,
    TranscriptionSegment as TranscriptionSegment,
    TranscriptionStreamResponse as TranscriptionStreamResponse,
    TranslationResponse as TranslationResponse,
    TranslationResponseStreamChoice as TranslationResponseStreamChoice,
    TranslationResponseVerbose as TranslationResponseVerbose,
    TranslationSegment as TranslationSegment,
    TranslationStreamResponse as TranslationStreamResponse,
)
from vllm.entrypoints.utils import get_max_tokens as get_max_tokens
from vllm.exceptions import VLLMValidationError as VLLMValidationError
from vllm.inputs import (
    EncoderDecoderInputs as EncoderDecoderInputs,
    ProcessorInputs as ProcessorInputs,
)
from vllm.logger import init_logger as init_logger
from vllm.logprobs import FlatLogprobs as FlatLogprobs, Logprob as Logprob
from vllm.model_executor.models import SupportsTranscription as SupportsTranscription
from vllm.multimodal.audio import split_audio as split_audio
from vllm.outputs import RequestOutput as RequestOutput
from vllm.renderers.inputs import (
    DictPrompt as DictPrompt,
    EncoderDecoderDictPrompt as EncoderDecoderDictPrompt,
)
from vllm.renderers.inputs.preprocess import (
    parse_enc_dec_prompt as parse_enc_dec_prompt,
    parse_model_prompt as parse_model_prompt,
)
from vllm.sampling_params import (
    BeamSearchParams as BeamSearchParams,
    SamplingParams as SamplingParams,
)
from vllm.tokenizers import get_tokenizer as get_tokenizer
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

SpeechToTextResponse: TypeAlias = TranscriptionResponse | TranslationResponse
SpeechToTextResponseVerbose: TypeAlias = (
    TranscriptionResponseVerbose | TranslationResponseVerbose
)
SpeechToTextSegment: TypeAlias = TranscriptionSegment | TranslationSegment
T = TypeVar("T", bound=SpeechToTextResponse)
V = TypeVar("V", bound=SpeechToTextResponseVerbose)
S = TypeVar("S", bound=SpeechToTextSegment)
ResponseType: TypeAlias = (
    TranscriptionResponse
    | TranslationResponse
    | TranscriptionResponseVerbose
    | TranslationResponseVerbose
)
logger: Incomplete

class OpenAISpeechToText(OpenAIServing):
    default_sampling_params: Incomplete
    task_type: Final[Incomplete]
    asr_config: Incomplete
    enable_force_include_usage: Incomplete
    max_audio_filesize_mb: Incomplete
    tokenizer: Incomplete
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        task_type: Literal["transcribe", "translate"] = "transcribe",
        enable_force_include_usage: bool = False,
    ) -> None: ...
    @cached_property
    def model_cls(self) -> type[SupportsTranscription]: ...
