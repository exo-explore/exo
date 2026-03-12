from _typeshed import Incomplete
from argparse import Namespace
from fastapi import FastAPI as FastAPI, WebSocket as WebSocket
from starlette.datastructures import State as State
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.realtime.connection import (
    RealtimeConnection as RealtimeConnection,
)
from vllm.entrypoints.openai.realtime.serving import (
    OpenAIServingRealtime as OpenAIServingRealtime,
)
from vllm.logger import init_logger as init_logger
from vllm.tasks import SupportedTask as SupportedTask

logger: Incomplete
router: Incomplete

async def realtime_endpoint(websocket: WebSocket): ...
def attach_router(app: FastAPI): ...
def init_realtime_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
): ...
