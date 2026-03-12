import abc
import enum
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput as ECConnectorOutput
from vllm.v1.request import Request as Request

logger: Incomplete

class ECConnectorRole(enum.Enum):
    SCHEDULER = 0
    WORKER = 1

class ECConnectorMetadata(ABC): ...

class ECConnectorBase(ABC, metaclass=abc.ABCMeta):
    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole) -> None: ...
    @property
    def role(self) -> ECConnectorRole: ...
    @property
    def is_producer(self) -> bool: ...
    @property
    def is_consumer(self) -> bool: ...
    def bind_connector_metadata(
        self, connector_metadata: ECConnectorMetadata
    ) -> None: ...
    def clear_connector_metadata(self) -> None: ...
    def register_caches(self, ec_caches: dict[str, torch.Tensor]): ...
    @abstractmethod
    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None: ...
    @abstractmethod
    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None: ...
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]: ...
    @abstractmethod
    def has_cache_item(self, identifier: str) -> bool: ...
    @abstractmethod
    def update_state_after_alloc(self, request: Request, index: int): ...
    @abstractmethod
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata: ...
    def update_connector_output(self, connector_output: ECConnectorOutput): ...
    def request_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]: ...
