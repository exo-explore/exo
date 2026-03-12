from _typeshed import Incomplete
from collections.abc import Callable as Callable
from concurrent.futures import Future
from dataclasses import dataclass
from ray.actor import ActorHandle
from ray.util.placement_group import PlacementGroup as PlacementGroup
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.ray.ray_env import get_env_vars_to_copy as get_env_vars_to_copy
from vllm.utils.network_utils import (
    get_distributed_init_method as get_distributed_init_method,
    get_ip as get_ip,
    get_open_port as get_open_port,
)
from vllm.v1.core.sched.output import (
    GrammarOutput as GrammarOutput,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.engine import (
    ReconfigureDistributedRequest as ReconfigureDistributedRequest,
    ReconfigureRankType as ReconfigureRankType,
)
from vllm.v1.executor.abstract import Executor as Executor
from vllm.v1.executor.ray_utils import (
    FutureWrapper as FutureWrapper,
    RayWorkerWrapper as RayWorkerWrapper,
    initialize_ray_cluster as initialize_ray_cluster,
    ray as ray,
)
from vllm.v1.outputs import ModelRunnerOutput as ModelRunnerOutput

logger: Incomplete
COMPLETED_NONE_FUTURE: Future[ModelRunnerOutput | None]

@dataclass
class RayWorkerMetaData:
    worker: ActorHandle
    created_rank: int
    adjusted_rank: int = ...
    ip: str = ...

class RayDistributedExecutor(Executor):
    WORKER_SPECIFIC_ENV_VARS: Incomplete
    uses_ray: bool
    supports_pp: bool
    @property
    def max_concurrent_batches(self) -> int: ...
    forward_dag: Incomplete
    def shutdown(self) -> None: ...
    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None: ...
    scheduler_output: Incomplete
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]: ...
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]: ...
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> list[Any] | Future[list[Any]]: ...
    def __del__(self) -> None: ...
    def check_health(self) -> None: ...
