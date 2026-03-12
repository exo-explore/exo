import torch
from collections.abc import Callable as Callable, Iterator
from dataclasses import dataclass
from typing import Any
from vllm import envs as envs
from vllm.config.parallel import ParallelConfig as ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig as WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine as WeightTransferEngine,
    WeightTransferInitInfo as WeightTransferInitInfo,
    WeightTransferUpdateInfo as WeightTransferUpdateInfo,
)

@dataclass
class IPCTrainerSendWeightsArgs:
    mode: str
    llm_handle: Any = ...
    url: str | None = ...
    def __post_init__(self) -> None: ...

@dataclass
class IPCWeightTransferInitInfo(WeightTransferInitInfo): ...

@dataclass
class IPCWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    ipc_handles: list[dict[str, tuple[Callable, tuple]]] | None = ...
    ipc_handles_pickled: str | None = ...
    def __post_init__(self) -> None: ...

class IPCWeightTransferEngine(
    WeightTransferEngine[IPCWeightTransferInitInfo, IPCWeightTransferUpdateInfo]
):
    init_info_cls = IPCWeightTransferInitInfo
    update_info_cls = IPCWeightTransferUpdateInfo
    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None: ...
    def init_transfer_engine(self, init_info: IPCWeightTransferInitInfo) -> None: ...
    def receive_weights(
        self,
        update_info: IPCWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None: ...
    def shutdown(self) -> None: ...
    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | IPCTrainerSendWeightsArgs,
    ) -> None: ...
