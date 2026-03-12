from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Request as Request
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete
router: Incomplete

def models(request: Request) -> OpenAIServingModels: ...
async def show_available_models(raw_request: Request): ...
def attach_router(app: FastAPI): ...
