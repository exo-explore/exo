from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI
from vllm.logger import init_logger as init_logger

logger: Incomplete

def attach_router(app: FastAPI) -> None: ...
