import torch
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group as get_kv_transfer_group,
    has_kv_transfer_group as has_kv_transfer_group,
    kv_transfer_state as kv_transfer_state,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    copy_kv_blocks as copy_kv_blocks,
)
from vllm.forward_context import (
    get_forward_context as get_forward_context,
    is_forward_context_available as is_forward_context_available,
    set_forward_context as set_forward_context,
)
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT as EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput as KVConnectorOutput,
    ModelRunnerOutput as ModelRunnerOutput,
)

class KVConnector:
    def pre_forward(self, scheduler_output: SchedulerOutput) -> None: ...
    def post_forward(
        self, scheduler_output: SchedulerOutput, wait_for_save: bool = True
    ) -> KVConnectorOutput | None: ...
    def no_forward(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput: ...
    def set_disabled(self, disabled: bool) -> None: ...

class ActiveKVConnector(KVConnector):
    vllm_config: Incomplete
    kv_connector: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
    ) -> None: ...
    def pre_forward(self, scheduler_output: SchedulerOutput) -> None: ...
    def post_forward(
        self,
        scheduler_output: SchedulerOutput,
        wait_for_save: bool = True,
        clear_metadata: bool = True,
    ) -> KVConnectorOutput | None: ...
    def clear_metadata(self) -> None: ...
    def no_forward(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput: ...
    def set_disabled(self, disabled: bool) -> None: ...

NO_OP_KV_CONNECTOR: Incomplete

def get_kv_connector(
    vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
) -> KVConnector: ...
