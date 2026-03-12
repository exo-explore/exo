from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Request as Request
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.logger import init_logger as init_logger

logger: Incomplete
router: Incomplete

def engine_client(request: Request) -> EngineClient: ...
async def collective_rpc(raw_request: Request): ...
def attach_router(app: FastAPI): ...
