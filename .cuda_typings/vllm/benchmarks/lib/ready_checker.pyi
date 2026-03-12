import aiohttp
from .endpoint_request_func import (
    RequestFunc as RequestFunc,
    RequestFuncInput as RequestFuncInput,
    RequestFuncOutput as RequestFuncOutput,
)
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger

logger: Incomplete

async def wait_for_endpoint(
    request_func: RequestFunc,
    test_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    timeout_seconds: int = 600,
    retry_interval: int = 5,
) -> RequestFuncOutput: ...
