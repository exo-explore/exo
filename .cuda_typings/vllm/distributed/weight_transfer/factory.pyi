from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.config.parallel import ParallelConfig as ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig as WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine as WeightTransferEngine,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete

class WeightTransferEngineFactory:
    @classmethod
    def register_engine(
        cls,
        name: str,
        module_path_or_cls: str | type[WeightTransferEngine],
        class_name: str | None = None,
    ) -> None: ...
    @classmethod
    def create_engine(
        cls, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> WeightTransferEngine: ...
