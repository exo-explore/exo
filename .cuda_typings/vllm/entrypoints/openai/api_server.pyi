import asyncio
import socket
import tempfile
from _typeshed import Incomplete
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.datastructures import State as State
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs as AsyncEngineArgs
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.chat_utils import load_chat_template as load_chat_template
from vllm.entrypoints.launcher import serve_http as serve_http
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser as make_arg_parser,
    validate_parsed_serve_args as validate_parsed_serve_args,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath as BaseModelPath
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.openai.server_utils import (
    engine_error_handler as engine_error_handler,
    exception_handler as exception_handler,
    get_uvicorn_log_config as get_uvicorn_log_config,
    http_exception_handler as http_exception_handler,
    lifespan as lifespan,
    log_response as log_response,
    validation_exception_handler as validation_exception_handler,
)
from vllm.entrypoints.sagemaker.api_router import (
    sagemaker_standards_bootstrap as sagemaker_standards_bootstrap,
)
from vllm.entrypoints.serve.elastic_ep.middleware import (
    ScalingMiddleware as ScalingMiddleware,
)
from vllm.entrypoints.serve.tokenize.serving import (
    OpenAIServingTokenization as OpenAIServingTokenization,
)
from vllm.entrypoints.utils import (
    cli_env_setup as cli_env_setup,
    log_non_default_args as log_non_default_args,
    log_version_and_model as log_version_and_model,
    process_lora_modules as process_lora_modules,
)
from vllm.logger import init_logger as init_logger
from vllm.reasoning import ReasoningParserManager as ReasoningParserManager
from vllm.tasks import POOLING_TASKS as POOLING_TASKS, SupportedTask as SupportedTask
from vllm.tool_parsers import ToolParserManager as ToolParserManager
from vllm.tracing import instrument as instrument
from vllm.usage.usage_lib import UsageContext as UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser
from vllm.utils.network_utils import is_valid_ipv6_address as is_valid_ipv6_address
from vllm.utils.system_utils import (
    decorate_logs as decorate_logs,
    set_ulimit as set_ulimit,
)
from vllm.v1.engine.exceptions import (
    EngineDeadError as EngineDeadError,
    EngineGenerateError as EngineGenerateError,
)

prometheus_multiproc_dir: tempfile.TemporaryDirectory
logger: Incomplete

@asynccontextmanager
async def build_async_engine_client(
    args: Namespace,
    *,
    usage_context: UsageContext = ...,
    disable_frontend_multiprocessing: bool | None = None,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]: ...
@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    *,
    usage_context: UsageContext = ...,
    disable_frontend_multiprocessing: bool = False,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]: ...
def build_app(
    args: Namespace, supported_tasks: tuple["SupportedTask", ...] | None = None
) -> FastAPI: ...
async def init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple["SupportedTask", ...] | None = None,
) -> None: ...
async def init_render_app_state(
    vllm_config: VllmConfig, state: State, args: Namespace
) -> None: ...
def create_server_socket(addr: tuple[str, int]) -> socket.socket: ...
def create_server_unix_socket(path: str) -> socket.socket: ...
def validate_api_server_args(args) -> None: ...
def setup_server(args): ...
async def build_and_serve(
    engine_client: EngineClient,
    listen_address: str,
    sock: socket.socket,
    args: Namespace,
    **uvicorn_kwargs,
) -> asyncio.Task: ...
async def build_and_serve_renderer(
    vllm_config: VllmConfig,
    listen_address: str,
    sock: socket.socket,
    args: Namespace,
    **uvicorn_kwargs,
) -> asyncio.Task: ...
async def run_server(args, **uvicorn_kwargs) -> None: ...
async def run_server_worker(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None: ...
