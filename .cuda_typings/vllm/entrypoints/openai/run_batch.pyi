from _typeshed import Incomplete
from argparse import Namespace
from collections.abc import Awaitable, Callable
from fastapi import UploadFile
from pydantic_core.core_schema import ValidationInfo as ValidationInfo
from tqdm import tqdm
from typing import Any, TypeAlias
from vllm.config import config as config
from vllm.engine.arg_utils import AsyncEngineArgs as AsyncEngineArgs
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.openai.api_server import init_app_state as init_app_state
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
    ChatCompletionResponse as ChatCompletionResponse,
)
from vllm.entrypoints.openai.cli_args import BaseFrontendArgs as BaseFrontendArgs
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo as ErrorInfo,
    ErrorResponse as ErrorResponse,
    OpenAIBaseModel as OpenAIBaseModel,
)
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionRequest as TranscriptionRequest,
    TranscriptionResponse as TranscriptionResponse,
    TranscriptionResponseVerbose as TranscriptionResponseVerbose,
    TranslationRequest as TranslationRequest,
    TranslationResponse as TranslationResponse,
    TranslationResponseVerbose as TranslationResponseVerbose,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingRequest as EmbeddingRequest,
    EmbeddingResponse as EmbeddingResponse,
)
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest as RerankRequest,
    RerankResponse as RerankResponse,
    ScoreRequest as ScoreRequest,
    ScoreResponse as ScoreResponse,
)
from vllm.entrypoints.utils import create_error_response as create_error_response
from vllm.logger import init_logger as init_logger
from vllm.reasoning import ReasoningParserManager as ReasoningParserManager
from vllm.utils import random_uuid as random_uuid
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser

logger: Incomplete

class BatchTranscriptionRequest(TranscriptionRequest):
    file_url: str
    file: UploadFile | None
    @classmethod
    def validate_no_file(cls, data: Any): ...

class BatchTranslationRequest(TranslationRequest):
    file_url: str
    file: UploadFile | None
    @classmethod
    def validate_no_file(cls, data: Any): ...

BatchRequestInputBody: TypeAlias = (
    ChatCompletionRequest
    | EmbeddingRequest
    | ScoreRequest
    | RerankRequest
    | BatchTranscriptionRequest
    | BatchTranslationRequest
)

class BatchRequestInput(OpenAIBaseModel):
    custom_id: str
    method: str
    url: str
    body: BatchRequestInputBody
    @classmethod
    def check_type_for_url(cls, value: Any, info: ValidationInfo): ...

class BatchResponseData(OpenAIBaseModel):
    status_code: int
    request_id: str
    body: (
        ChatCompletionResponse
        | EmbeddingResponse
        | ScoreResponse
        | RerankResponse
        | TranscriptionResponse
        | TranscriptionResponseVerbose
        | TranslationResponse
        | TranslationResponseVerbose
        | None
    )

class BatchRequestOutput(OpenAIBaseModel):
    id: str
    custom_id: str
    response: BatchResponseData | None
    error: Any | None

@config
class BatchFrontendArgs(BaseFrontendArgs):
    input_file: str | None = ...
    output_file: str | None = ...
    output_tmp_dir: str | None = ...
    enable_metrics: bool = ...
    host: str | None = ...
    port: int = ...
    url: str = ...

def make_arg_parser(parser: FlexibleArgumentParser): ...
def parse_args(): ...

class BatchProgressTracker:
    def __init__(self) -> None: ...
    def submitted(self) -> None: ...
    def completed(self) -> None: ...
    def pbar(self) -> tqdm: ...

async def read_file(path_or_url: str) -> str: ...
async def write_local_file(
    output_path: str, batch_outputs: list[BatchRequestOutput]
) -> None: ...
async def upload_data(output_url: str, data_or_file: str, from_file: bool) -> None: ...
async def write_file(
    path_or_url: str, batch_outputs: list[BatchRequestOutput], output_tmp_dir: str
) -> None: ...
async def download_bytes_from_url(url: str) -> bytes: ...
def make_error_request_output(
    request: BatchRequestInput, error_msg: str
) -> BatchRequestOutput: ...
async def make_async_error_request_output(
    request: BatchRequestInput, error_msg: str
) -> BatchRequestOutput: ...
async def run_request(
    serving_engine_func: Callable,
    request: BatchRequestInput,
    tracker: BatchProgressTracker,
) -> BatchRequestOutput: ...

WrapperFn: TypeAlias = Callable[[Callable], Callable]

def handle_endpoint_request(
    request: BatchRequestInput,
    tracker: BatchProgressTracker,
    url_matcher: Callable[[str], bool],
    handler_getter: Callable[[], Callable | None],
    wrapper_fn: WrapperFn | None = None,
) -> Awaitable[BatchRequestOutput] | None: ...
def make_transcription_wrapper(is_translation: bool) -> WrapperFn: ...
async def build_endpoint_registry(
    engine_client: EngineClient, args: Namespace
) -> dict[str, dict[str, Any]]: ...
def validate_run_batch_args(args) -> None: ...
async def run_batch(engine_client: EngineClient, args: Namespace) -> None: ...
async def main(args: Namespace): ...
