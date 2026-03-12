from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.metrics.stats import SchedulerStats as SchedulerStats

logger: Incomplete

def prepare_object_to_dump(obj) -> str: ...
def dump_engine_exception(
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
): ...
