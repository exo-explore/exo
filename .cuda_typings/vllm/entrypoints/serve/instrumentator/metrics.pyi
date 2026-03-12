from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Response
from vllm.v1.metrics.prometheus import (
    get_prometheus_registry as get_prometheus_registry,
)

class PrometheusResponse(Response):
    media_type: Incomplete

def attach_router(app: FastAPI): ...
