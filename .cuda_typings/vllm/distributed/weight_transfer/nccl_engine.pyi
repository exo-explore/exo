import torch
from collections.abc import Callable as Callable, Iterator
from dataclasses import dataclass
from typing import Any
from vllm.config.parallel import ParallelConfig as ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig as WeightTransferConfig
from vllm.distributed.device_communicators.pynccl import (
    PyNcclCommunicator as PyNcclCommunicator,
)
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine as WeightTransferEngine,
    WeightTransferInitInfo as WeightTransferInitInfo,
    WeightTransferUpdateInfo as WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES as DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS as DEFAULT_PACKED_NUM_BUFFERS,
    packed_broadcast_consumer as packed_broadcast_consumer,
)

@dataclass
class NCCLWeightTransferInitInfo(WeightTransferInitInfo):
    master_address: str
    master_port: int
    rank_offset: int
    world_size: int

@dataclass
class NCCLTrainerSendWeightsArgs:
    group: Any
    src: int = ...
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor] | None = ...
    packed: bool = ...
    stream: torch.cuda.Stream | None = ...
    packed_buffer_size_bytes: int = ...
    packed_num_buffers: int = ...

@dataclass
class NCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    packed: bool = ...
    packed_buffer_size_bytes: int = ...
    packed_num_buffers: int = ...
    def __post_init__(self) -> None: ...

class NCCLWeightTransferEngine(
    WeightTransferEngine[NCCLWeightTransferInitInfo, NCCLWeightTransferUpdateInfo]
):
    init_info_cls = NCCLWeightTransferInitInfo
    update_info_cls = NCCLWeightTransferUpdateInfo
    model_update_group: PyNcclCommunicator | None
    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None: ...
    def init_transfer_engine(self, init_info: NCCLWeightTransferInitInfo) -> None: ...
    def receive_weights(
        self,
        update_info: NCCLWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None: ...
    def shutdown(self) -> None: ...
    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | NCCLTrainerSendWeightsArgs,
    ) -> None: ...
    @staticmethod
    def trainer_init(
        init_info: NCCLWeightTransferInitInfo | dict,
    ) -> PyNcclCommunicator: ...
