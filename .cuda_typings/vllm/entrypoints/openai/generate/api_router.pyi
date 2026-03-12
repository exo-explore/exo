from argparse import Namespace
from fastapi import FastAPI as FastAPI
from starlette.datastructures import State as State
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.tasks import SupportedTask as SupportedTask

def register_generate_api_routers(app: FastAPI): ...
async def init_generate_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
): ...
