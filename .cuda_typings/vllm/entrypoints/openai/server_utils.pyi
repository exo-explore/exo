from _typeshed import Incomplete
from argparse import Namespace
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI as FastAPI,
    HTTPException as HTTPException,
    Request as Request,
)
from fastapi.exceptions import RequestValidationError as RequestValidationError
from starlette.datastructures import Headers
from starlette.types import (
    ASGIApp as ASGIApp,
    Message as Message,
    Receive as Receive,
    Scope as Scope,
    Send as Send,
)
from vllm import envs as envs
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.launcher import terminate_if_errored as terminate_if_errored
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo as ErrorInfo,
    ErrorResponse as ErrorResponse,
)
from vllm.entrypoints.utils import (
    create_error_response as create_error_response,
    sanitize_message as sanitize_message,
)
from vllm.exceptions import VLLMValidationError as VLLMValidationError
from vllm.logger import init_logger as init_logger
from vllm.utils.gc_utils import freeze_gc_heap as freeze_gc_heap
from vllm.v1.engine.exceptions import (
    EngineDeadError as EngineDeadError,
    EngineGenerateError as EngineGenerateError,
)

logger: Incomplete

class AuthenticationMiddleware:
    app: Incomplete
    api_tokens: Incomplete
    def __init__(self, app: ASGIApp, tokens: list[str]) -> None: ...
    def verify_token(self, headers: Headers) -> bool: ...
    def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> Awaitable[None]: ...

class XRequestIdMiddleware:
    app: Incomplete
    def __init__(self, app: ASGIApp) -> None: ...
    def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> Awaitable[None]: ...

def load_log_config(log_config_file: str | None) -> dict | None: ...
def get_uvicorn_log_config(args: Namespace) -> dict | None: ...

class SSEDecoder:
    buffer: str
    content_buffer: Incomplete
    def __init__(self) -> None: ...
    def decode_chunk(self, chunk: bytes) -> list[dict]: ...
    def extract_content(self, event_data: dict) -> str: ...
    def add_content(self, content: str) -> None: ...
    def get_complete_content(self) -> str: ...

async def log_response(request: Request, call_next): ...
async def engine_error_handler(
    req: Request, exc: EngineDeadError | EngineGenerateError
): ...
async def exception_handler(req: Request, exc: Exception): ...
async def http_exception_handler(req: Request, exc: HTTPException): ...
async def validation_exception_handler(req: Request, exc: RequestValidationError): ...
@asynccontextmanager
async def lifespan(app: FastAPI): ...
