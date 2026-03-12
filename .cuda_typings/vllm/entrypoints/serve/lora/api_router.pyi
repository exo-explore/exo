from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Request as Request
from vllm import envs as envs
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.models.api_router import models as models
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest as LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest as UnloadLoRAAdapterRequest,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete
router: Incomplete

def attach_router(app: FastAPI): ...
