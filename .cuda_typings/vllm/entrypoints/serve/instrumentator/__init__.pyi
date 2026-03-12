from fastapi import FastAPI as FastAPI
from vllm import envs as envs

def register_instrumentator_api_routers(app: FastAPI): ...
