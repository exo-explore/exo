from collections.abc import Awaitable, Callable
from fastapi import FastAPI as FastAPI, Request
from typing import Any
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing as OpenAIServing
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.pooling.base.serving import PoolingServing as PoolingServing
from vllm.entrypoints.serve.instrumentator.basic import base as base
from vllm.entrypoints.serve.instrumentator.health import health as health
from vllm.tasks import POOLING_TASKS as POOLING_TASKS, SupportedTask as SupportedTask

RequestType = Any
GetHandlerFn = Callable[[Request], OpenAIServing | PoolingServing | None]
EndpointFn = Callable[[RequestType, Request], Awaitable[Any]]

def get_invocation_types(supported_tasks: tuple["SupportedTask", ...]): ...
def attach_router(app: FastAPI, supported_tasks: tuple["SupportedTask", ...]): ...
def sagemaker_standards_bootstrap(app: FastAPI) -> FastAPI: ...
