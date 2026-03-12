from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Request as Request
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest as CompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.serve.render.serving import (
    OpenAIServingRender as OpenAIServingRender,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete
router: Incomplete

def render(request: Request) -> OpenAIServingRender | None: ...
async def render_chat_completion(
    request: ChatCompletionRequest, raw_request: Request
): ...
async def render_completion(request: CompletionRequest, raw_request: Request): ...
def attach_router(app: FastAPI) -> None: ...
