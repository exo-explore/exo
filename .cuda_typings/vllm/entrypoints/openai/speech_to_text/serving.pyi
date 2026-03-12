from _typeshed import Incomplete
from collections.abc import AsyncGenerator
from fastapi import Request as Request
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as ErrorResponse,
    RequestResponseMetadata as RequestResponseMetadata,
)
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionRequest as TranscriptionRequest,
    TranscriptionResponse as TranscriptionResponse,
    TranscriptionResponseStreamChoice as TranscriptionResponseStreamChoice,
    TranscriptionResponseVerbose as TranscriptionResponseVerbose,
    TranscriptionStreamResponse as TranscriptionStreamResponse,
    TranslationRequest as TranslationRequest,
    TranslationResponse as TranslationResponse,
    TranslationResponseStreamChoice as TranslationResponseStreamChoice,
    TranslationResponseVerbose as TranslationResponseVerbose,
    TranslationStreamResponse as TranslationStreamResponse,
)
from vllm.entrypoints.openai.speech_to_text.speech_to_text import (
    OpenAISpeechToText as OpenAISpeechToText,
)
from vllm.logger import init_logger as init_logger
from vllm.outputs import RequestOutput as RequestOutput

logger: Incomplete

class OpenAIServingTranscription(OpenAISpeechToText):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        enable_force_include_usage: bool = False,
    ) -> None: ...
    async def create_transcription(
        self,
        audio_data: bytes,
        request: TranscriptionRequest,
        raw_request: Request | None = None,
    ) -> (
        TranscriptionResponse
        | TranscriptionResponseVerbose
        | AsyncGenerator[str, None]
        | ErrorResponse
    ): ...
    async def transcription_stream_generator(
        self,
        request: TranscriptionRequest,
        result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
    ) -> AsyncGenerator[str, None]: ...

class OpenAIServingTranslation(OpenAISpeechToText):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        enable_force_include_usage: bool = False,
    ) -> None: ...
    async def create_translation(
        self,
        audio_data: bytes,
        request: TranslationRequest,
        raw_request: Request | None = None,
    ) -> (
        TranslationResponse
        | TranslationResponseVerbose
        | AsyncGenerator[str, None]
        | ErrorResponse
    ): ...
    async def translation_stream_generator(
        self,
        request: TranslationRequest,
        result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
    ) -> AsyncGenerator[str, None]: ...
