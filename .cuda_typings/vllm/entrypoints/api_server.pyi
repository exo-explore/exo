from _typeshed import Incomplete
from argparse import Namespace
from fastapi import FastAPI, Request as Request
from fastapi.responses import Response
from typing import Any
from vllm.engine.arg_utils import AsyncEngineArgs as AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine as AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http as serve_http
from vllm.entrypoints.utils import with_cancellation as with_cancellation
from vllm.logger import init_logger as init_logger
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.usage.usage_lib import UsageContext as UsageContext
from vllm.utils import random_uuid as random_uuid
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit as set_ulimit

logger: Incomplete
app: Incomplete
engine: Incomplete

async def health() -> Response: ...
async def generate(request: Request) -> Response: ...
def build_app(args: Namespace) -> FastAPI: ...
async def init_app(
    args: Namespace, llm_engine: AsyncLLMEngine | None = None
) -> FastAPI: ...
async def run_server(
    args: Namespace, llm_engine: AsyncLLMEngine | None = None, **uvicorn_kwargs: Any
) -> None: ...
