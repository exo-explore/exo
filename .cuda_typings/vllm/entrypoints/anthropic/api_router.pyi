from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Request as Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest as AnthropicCountTokensRequest,
    AnthropicCountTokensResponse as AnthropicCountTokensResponse,
    AnthropicError as AnthropicError,
    AnthropicErrorResponse as AnthropicErrorResponse,
    AnthropicMessagesRequest as AnthropicMessagesRequest,
    AnthropicMessagesResponse as AnthropicMessagesResponse,
)
from vllm.entrypoints.anthropic.serving import (
    AnthropicServingMessages as AnthropicServingMessages,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call as load_aware_call,
    with_cancellation as with_cancellation,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete
router: Incomplete

def messages(request: Request) -> AnthropicServingMessages: ...
def translate_error_response(response: ErrorResponse) -> JSONResponse: ...
@with_cancellation
@load_aware_call
async def create_messages(request: AnthropicMessagesRequest, raw_request: Request): ...
@load_aware_call
@with_cancellation
async def count_tokens(request: AnthropicCountTokensRequest, raw_request: Request): ...
def attach_router(app: FastAPI): ...
