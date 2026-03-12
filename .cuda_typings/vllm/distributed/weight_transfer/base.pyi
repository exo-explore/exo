import abc
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable as Callable, Iterator
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Generic, TypeVar
from vllm.config.parallel import ParallelConfig as ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig as WeightTransferConfig

TInitInfo = TypeVar("TInitInfo", bound="WeightTransferInitInfo")
TUpdateInfo = TypeVar("TUpdateInfo", bound="WeightTransferUpdateInfo")

@dataclass
class WeightTransferInitInfo(ABC): ...

@dataclass
class WeightTransferUpdateInfo(ABC):
    _: KW_ONLY
    is_checkpoint_format: bool = ...

@dataclass
class WeightTransferInitRequest:
    init_info: dict[str, Any] = field(default_factory=dict)

@dataclass
class WeightTransferUpdateRequest:
    update_info: dict[str, Any] = field(default_factory=dict)

class WeightTransferEngine(ABC, Generic[TInitInfo, TUpdateInfo], metaclass=abc.ABCMeta):
    init_info_cls: type[TInitInfo]
    update_info_cls: type[TUpdateInfo]
    config: Incomplete
    parallel_config: Incomplete
    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None: ...
    def parse_init_info(self, init_dict: dict[str, Any]) -> TInitInfo: ...
    def parse_update_info(self, update_dict: dict[str, Any]) -> TUpdateInfo: ...
    @abstractmethod
    def init_transfer_engine(self, init_info: TInitInfo) -> None: ...
    @abstractmethod
    def receive_weights(
        self,
        update_info: TUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None: ...
    @abstractmethod
    def shutdown(self) -> None: ...
    @staticmethod
    @abstractmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]], trainer_args: dict[str, Any] | Any
    ) -> None: ...
