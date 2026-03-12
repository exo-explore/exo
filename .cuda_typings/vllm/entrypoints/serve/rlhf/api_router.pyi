from _typeshed import Incomplete
from fastapi import FastAPI as FastAPI, Request as Request
from fastapi.responses import JSONResponse
from typing import Annotated
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest as WeightTransferInitRequest,
    WeightTransferUpdateRequest as WeightTransferUpdateRequest,
)
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.logger import init_logger as init_logger
from vllm.v1.engine import PauseMode as PauseMode

logger: Incomplete

def engine_client(request: Request) -> EngineClient: ...

router: Incomplete

async def pause_generation(
    raw_request: Request,
    mode: Annotated[PauseMode, None] = "abort",
    wait_for_inflight_requests: bool = ...,
    clear_cache: Annotated[bool, None] = True,
) -> JSONResponse: ...
async def resume_generation(raw_request: Request) -> JSONResponse: ...
async def is_paused(raw_request: Request) -> JSONResponse: ...
async def init_weight_transfer_engine(raw_request: Request): ...
async def update_weights(raw_request: Request): ...
async def get_world_size(raw_request: Request, include_dp: bool = ...): ...
def attach_router(app: FastAPI): ...
