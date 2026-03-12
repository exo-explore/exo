from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Request as Request
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.serve.elastic_ep.middleware import (
    get_scaling_elastic_ep as get_scaling_elastic_ep,
    set_scaling_elastic_ep as set_scaling_elastic_ep,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete

def engine_client(request: Request) -> EngineClient: ...

router: Incomplete

async def scale_elastic_ep(raw_request: Request): ...
async def is_scaling_elastic_ep(raw_request: Request): ...
def attach_router(app: FastAPI): ...
