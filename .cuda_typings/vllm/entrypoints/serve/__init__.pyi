from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI
from vllm.logger import init_logger as init_logger

logger: Incomplete

def register_vllm_serve_api_routers(app: FastAPI): ...
