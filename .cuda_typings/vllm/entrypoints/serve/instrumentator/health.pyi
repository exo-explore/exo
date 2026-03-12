from _typeshed import Incomplete
from fastapi import Request as Request
from fastapi.responses import Response
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.logger import init_logger as init_logger
from vllm.v1.engine.exceptions import EngineDeadError as EngineDeadError

logger: Incomplete
router: Incomplete

def engine_client(request: Request) -> EngineClient: ...
async def health(raw_request: Request) -> Response: ...
