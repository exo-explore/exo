from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Request as Request
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest as ResponsesRequest,
    ResponsesResponse as ResponsesResponse,
    StreamingResponsesResponse as StreamingResponsesResponse,
)
from vllm.entrypoints.openai.responses.serving import (
    OpenAIServingResponses as OpenAIServingResponses,
)
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call as load_aware_call,
    with_cancellation as with_cancellation,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete
router: Incomplete

def responses(request: Request) -> OpenAIServingResponses | None: ...
@with_cancellation
@load_aware_call
async def create_responses(request: ResponsesRequest, raw_request: Request): ...
@load_aware_call
async def retrieve_responses(
    response_id: str,
    raw_request: Request,
    starting_after: int | None = None,
    stream: bool | None = False,
): ...
@load_aware_call
async def cancel_responses(response_id: str, raw_request: Request): ...
def attach_router(app: FastAPI): ...
