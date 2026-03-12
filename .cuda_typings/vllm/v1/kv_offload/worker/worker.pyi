import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from dataclasses import dataclass
from vllm.logger import init_logger as init_logger
from vllm.v1.kv_offload.abstract import LoadStoreSpec as LoadStoreSpec

TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]
TransferType = tuple[str, str]
logger: Incomplete

@dataclass
class TransferResult:
    job_id: int
    success: bool
    transfer_size: int | None = ...
    transfer_time: float | None = ...
    transfer_type: TransferType | None = ...

class OffloadingHandler(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool: ...
    @abstractmethod
    def get_finished(self) -> list[TransferResult]: ...
    @abstractmethod
    def wait(self, job_ids: set[int]) -> None: ...

class OffloadingWorker:
    handlers: set[OffloadingHandler]
    transfer_type_to_handler: dict[TransferType, OffloadingHandler]
    def __init__(self) -> None: ...
    def register_handler(
        self,
        src_cls: type[LoadStoreSpec],
        dst_cls: type[LoadStoreSpec],
        handler: OffloadingHandler,
    ) -> None: ...
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool: ...
    def get_finished(self) -> list[TransferResult]: ...
    def wait(self, job_ids: set[int]) -> None: ...
