from _typeshed import Incomplete
from collections.abc import Callable as Callable
from concurrent.futures import Future
from functools import cached_property as cached_property
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.utils.network_utils import (
    get_distributed_init_method as get_distributed_init_method,
    get_ip as get_ip,
    get_open_port as get_open_port,
)
from vllm.v1.core.sched.output import (
    GrammarOutput as GrammarOutput,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.executor.abstract import Executor as Executor
from vllm.v1.outputs import (
    AsyncModelRunnerOutput as AsyncModelRunnerOutput,
    DraftTokenIds as DraftTokenIds,
    ModelRunnerOutput as ModelRunnerOutput,
)
from vllm.v1.serial_utils import run_method as run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase as WorkerWrapperBase

logger: Incomplete

class UniProcExecutor(Executor):
    @cached_property
    def max_concurrent_batches(self) -> int: ...
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any: ...
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]: ...
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]: ...
    def take_draft_token_ids(self) -> DraftTokenIds | None: ...
    def check_health(self) -> None: ...
    def shutdown(self) -> None: ...

class ExecutorWithExternalLauncher(UniProcExecutor):
    def determine_available_memory(self) -> list[int]: ...
