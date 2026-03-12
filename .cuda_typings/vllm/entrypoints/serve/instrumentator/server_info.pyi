from _typeshed import Incomplete
from fastapi import Query as Query, Request as Request
from typing import Annotated, Literal
from vllm.collect_env import get_env_info as get_env_info
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger

logger: Incomplete
router: Incomplete
PydanticVllmConfig: Incomplete

async def show_server_info(
    raw_request: Request,
    config_format: Annotated[Literal["text", "json"], None] = "text",
): ...
