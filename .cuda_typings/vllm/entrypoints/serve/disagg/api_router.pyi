from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Request as Request
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.serve.disagg.protocol import (
    GenerateRequest as GenerateRequest,
    GenerateResponse as GenerateResponse,
)
from vllm.entrypoints.serve.disagg.serving import ServingTokens as ServingTokens
from vllm.entrypoints.serve.tokenize.serving import (
    OpenAIServingTokenization as OpenAIServingTokenization,
)
from vllm.entrypoints.utils import (
    load_aware_call as load_aware_call,
    with_cancellation as with_cancellation,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete

def tokenization(request: Request) -> OpenAIServingTokenization: ...
def generate_tokens(request: Request) -> ServingTokens | None: ...
def engine_client(request: Request) -> EngineClient: ...

router: Incomplete

@with_cancellation
@load_aware_call
async def generate(request: GenerateRequest, raw_request: Request): ...
def attach_router(app: FastAPI): ...
