from _typeshed import Incomplete
from fastapi import Request as Request
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorResponse as IOProcessorResponse,
    PoolingBytesResponse as PoolingBytesResponse,
    PoolingRequest as PoolingRequest,
    PoolingResponse as PoolingResponse,
)
from vllm.entrypoints.pooling.pooling.serving import (
    OpenAIServingPooling as OpenAIServingPooling,
)
from vllm.entrypoints.utils import (
    load_aware_call as load_aware_call,
    with_cancellation as with_cancellation,
)

router: Incomplete

def pooling(request: Request) -> OpenAIServingPooling | None: ...
@with_cancellation
@load_aware_call
async def create_pooling(request: PoolingRequest, raw_request: Request): ...
