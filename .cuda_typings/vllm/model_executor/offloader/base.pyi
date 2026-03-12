import abc
import torch.nn as nn
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Generator
from vllm.config import OffloadConfig as OffloadConfig
from vllm.logger import init_logger as init_logger

logger: Incomplete

class BaseOffloader(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def wrap_modules(
        self, modules_generator: Generator[nn.Module, None, None]
    ) -> list[nn.Module]: ...
    def post_init(self) -> None: ...
    def sync_prev_onload(self) -> None: ...
    def join_after_forward(self) -> None: ...

class NoopOffloader(BaseOffloader):
    def wrap_modules(
        self, modules_generator: Generator[nn.Module, None, None]
    ) -> list[nn.Module]: ...

def get_offloader() -> BaseOffloader: ...
def set_offloader(instance: BaseOffloader) -> None: ...
def create_offloader(offload_config: OffloadConfig) -> BaseOffloader: ...
