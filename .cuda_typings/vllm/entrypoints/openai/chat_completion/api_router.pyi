from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Request as Request
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
    ChatCompletionResponse as ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import (
    OpenAIServingChat as OpenAIServingChat,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.orca_metrics import metrics_header as metrics_header
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call as load_aware_call,
    with_cancellation as with_cancellation,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete
router: Incomplete
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL: str

def chat(request: Request) -> OpenAIServingChat | None: ...
@with_cancellation
@load_aware_call
async def create_chat_completion(
    request: ChatCompletionRequest, raw_request: Request
): ...
def attach_router(app: FastAPI): ...
