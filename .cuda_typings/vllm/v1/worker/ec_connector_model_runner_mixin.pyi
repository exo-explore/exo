import torch
from _typeshed import Incomplete
from contextlib import AbstractContextManager
from vllm.distributed.ec_transfer import (
    get_ec_transfer as get_ec_transfer,
    has_ec_transfer as has_ec_transfer,
)
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase as ECConnectorBase,
)
from vllm.logger import init_logger as init_logger
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput as ECConnectorOutput

logger: Incomplete

class ECConnectorModelRunnerMixin:
    @staticmethod
    def maybe_save_ec_to_connector(
        encoder_cache: dict[str, torch.Tensor], mm_hash: str
    ): ...
    @staticmethod
    def get_finished_ec_transfers(
        scheduler_output: SchedulerOutput,
    ) -> tuple[set[str] | None, set[str] | None]: ...
    @staticmethod
    def maybe_get_ec_connector_output(
        scheduler_output: SchedulerOutput,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> AbstractContextManager[ECConnectorOutput | None]: ...
