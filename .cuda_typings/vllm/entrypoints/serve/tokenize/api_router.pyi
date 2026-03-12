from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Request as Request
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest as DetokenizeRequest,
    DetokenizeResponse as DetokenizeResponse,
    TokenizeRequest as TokenizeRequest,
    TokenizeResponse as TokenizeResponse,
)
from vllm.entrypoints.serve.tokenize.serving import (
    OpenAIServingTokenization as OpenAIServingTokenization,
)
from vllm.entrypoints.utils import with_cancellation as with_cancellation
from vllm.logger import init_logger as init_logger

logger: Incomplete

def tokenization(request: Request) -> OpenAIServingTokenization: ...

router: Incomplete

@with_cancellation
async def tokenize(request: TokenizeRequest, raw_request: Request): ...
@with_cancellation
async def detokenize(request: DetokenizeRequest, raw_request: Request): ...
def attach_router(app: FastAPI): ...
