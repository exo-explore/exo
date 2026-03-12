import enum
from _typeshed import Incomplete
from typing import Literal, TypeAlias
from vllm.config import ParallelConfig as ParallelConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    sched_yield as sched_yield,
    stateless_destroy_torch_distributed_process_group as stateless_destroy_torch_distributed_process_group,
)
from vllm.logger import init_logger as init_logger
from vllm.v1.engine import (
    EEPNotificationType as EEPNotificationType,
    ReconfigureDistributedRequest as ReconfigureDistributedRequest,
    ReconfigureRankType as ReconfigureRankType,
)
from vllm.v1.engine.core import DPEngineCoreProc as DPEngineCoreProc
from vllm.v1.executor.abstract import Executor as Executor

logger: Incomplete
WorkerType: Incomplete

class ScaleUpExistingEngineState(enum.IntEnum):
    WAIT_NEW_CORE_ENGINES_INIT = 0
    CREATE_STANDBY_GROUPS = 1
    TRANSFER_EXPERT_MAPPING = 2
    WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT = 3
    TRANSFER_WEIGHTS = 4
    SYNC_KV_CACHE_MEMORY_SIZE = 5
    SWITCH_AND_PREPARE = 6
    EPLB_RESHUFFLE = 7
    COMPLETE = 8

class ScaleUpNewEngineState(enum.IntEnum):
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    COMPLETE = 2

class ScaleDownRemainingEngineState(enum.IntEnum):
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    SWITCH_AND_PREPARE = 2
    COMPLETE = 3

class ScaleDownRemovingEngineState(enum.IntEnum):
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    COMPLETE = 2

EngineState: TypeAlias = (
    ScaleUpExistingEngineState
    | ScaleUpNewEngineState
    | ScaleDownRemainingEngineState
    | ScaleDownRemovingEngineState
)

class _BarrierTimeoutError(RuntimeError): ...

class ElasticEPScalingState:
    model_executor_ref: Incomplete
    engine_core_ref: Incomplete
    vllm_config: Incomplete
    old_dp_group: Incomplete
    old_dp_store: Incomplete
    new_parallel_config: ParallelConfig
    new_dp_group: Incomplete
    new_dp_store: Incomplete
    worker_type: Incomplete
    scale_type: Incomplete
    reconfig_request: Incomplete
    state: EngineState
    def __init__(
        self,
        model_executor: Executor,
        engine_core: DPEngineCoreProc,
        vllm_config: VllmConfig,
        new_parallel_config: ParallelConfig,
        worker_type: WorkerType,
        scale_type: Literal["scale_up", "scale_down"],
        reconfig_request: ReconfigureDistributedRequest | None = None,
    ) -> None: ...
    @property
    def model_executor(self) -> Executor: ...
    @property
    def engine_core(self) -> DPEngineCoreProc: ...
    def progress(self) -> bool: ...
    def handle_notification(self, notification_type: EEPNotificationType): ...
    def is_complete(self) -> bool: ...
