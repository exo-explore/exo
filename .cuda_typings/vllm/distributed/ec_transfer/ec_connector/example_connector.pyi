from _typeshed import Incomplete
from dataclasses import dataclass
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase as ECConnectorBase,
    ECConnectorMetadata as ECConnectorMetadata,
    ECConnectorRole as ECConnectorRole,
)
from vllm.logger import init_logger as init_logger
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.request import Request as Request

logger: Incomplete

@dataclass
class MMMeta:
    mm_hash: str
    num_token: int
    @staticmethod
    def make_meta(mm_hash, num_token) -> MMMeta: ...

@dataclass
class ECExampleConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]
    def __init__(self) -> None: ...
    def add_mm_data(self, mm_data: MMMeta): ...

class ECExampleConnector(ECConnectorBase):
    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole) -> None: ...
    def start_load_caches(self, encoder_cache, **kwargs) -> None: ...
    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None: ...
    def has_cache_item(self, identifier: str) -> bool: ...
    def update_state_after_alloc(self, request: Request, index: int) -> None: ...
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata: ...
