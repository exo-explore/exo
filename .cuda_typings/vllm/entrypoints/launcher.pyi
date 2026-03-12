import socket
import uvicorn
from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI
from typing import Any
from vllm import envs as envs
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT as H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT as H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from vllm.entrypoints.ssl import SSLCertRefresher as SSLCertRefresher
from vllm.logger import init_logger as init_logger
from vllm.utils.network_utils import find_process_using_port as find_process_using_port

logger: Incomplete

async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    **uvicorn_kwargs: Any,
): ...
async def watchdog_loop(server: uvicorn.Server, engine: EngineClient): ...
def terminate_if_errored(server: uvicorn.Server, engine: EngineClient): ...
