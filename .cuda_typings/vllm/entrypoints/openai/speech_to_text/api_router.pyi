from _typeshed import Incomplete
from argparse import Namespace
from fastapi import FastAPI as FastAPI, Form as Form, Request as Request
from starlette.datastructures import State as State
from typing import Annotated
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionRequest as TranscriptionRequest,
    TranscriptionResponseVariant as TranscriptionResponseVariant,
    TranslationRequest as TranslationRequest,
    TranslationResponseVariant as TranslationResponseVariant,
)
from vllm.entrypoints.openai.speech_to_text.serving import (
    OpenAIServingTranscription as OpenAIServingTranscription,
    OpenAIServingTranslation as OpenAIServingTranslation,
)
from vllm.entrypoints.utils import (
    load_aware_call as load_aware_call,
    with_cancellation as with_cancellation,
)
from vllm.logger import init_logger as init_logger
from vllm.tasks import SupportedTask as SupportedTask

logger: Incomplete
router: Incomplete

def transcription(request: Request) -> OpenAIServingTranscription: ...
def translation(request: Request) -> OpenAIServingTranslation: ...
@with_cancellation
@load_aware_call
async def create_transcriptions(
    raw_request: Request, request: Annotated[TranscriptionRequest, None]
): ...
@with_cancellation
@load_aware_call
async def create_translations(
    request: Annotated[TranslationRequest, None], raw_request: Request
): ...
def attach_router(app: FastAPI): ...
def init_transcription_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
): ...
