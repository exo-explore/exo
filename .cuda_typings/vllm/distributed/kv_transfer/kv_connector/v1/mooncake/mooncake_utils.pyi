import threading
import uvicorn
from _typeshed import Incomplete
from dataclasses import dataclass
from pydantic import BaseModel
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId as EngineId
from vllm.logger import init_logger as init_logger

WorkerAddr = str
logger: Incomplete

class RegisterWorkerPayload(BaseModel):
    engine_id: EngineId
    dp_rank: int
    tp_rank: int
    pp_rank: int
    addr: WorkerAddr

@dataclass
class EngineEntry:
    engine_id: EngineId
    worker_addr: dict[int, dict[int, WorkerAddr]]

class MooncakeBootstrapServer:
    workers: dict[int, EngineEntry]
    host: Incomplete
    port: Incomplete
    app: Incomplete
    server_thread: threading.Thread | None
    server: uvicorn.Server | None
    def __init__(self, vllm_config: VllmConfig, host: str, port: int) -> None: ...
    def __del__(self) -> None: ...
    def start(self) -> None: ...
    def shutdown(self) -> None: ...
    async def register_worker(self, payload: RegisterWorkerPayload): ...
    async def query(self) -> dict[int, EngineEntry]: ...
