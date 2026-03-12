from _typeshed import Incomplete
from fastapi import Request as Request
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.openai.engine.serving import OpenAIServing as OpenAIServing
from vllm.entrypoints.serve.tokenize.serving import (
    OpenAIServingTokenization as OpenAIServingTokenization,
)
from vllm.logger import init_logger as init_logger

router: Incomplete
logger: Incomplete

def base(request: Request) -> OpenAIServing: ...
def tokenization(request: Request) -> OpenAIServingTokenization: ...
def engine_client(request: Request) -> EngineClient: ...
async def get_server_load_metrics(request: Request): ...
async def show_version(): ...
