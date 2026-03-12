from _typeshed import Incomplete
from concurrent.futures import Future
from ray.util.placement_group import PlacementGroup as PlacementGroup
from vllm.config import ParallelConfig as ParallelConfig
from vllm.distributed import get_pp_group as get_pp_group
from vllm.distributed.kv_transfer.kv_connector.utils import (
    KVOutputAggregator as KVOutputAggregator,
)
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.network_utils import get_ip as get_ip
from vllm.v1.core.sched.output import (
    GrammarOutput as GrammarOutput,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.outputs import (
    AsyncModelRunnerOutput as AsyncModelRunnerOutput,
    ModelRunnerOutput as ModelRunnerOutput,
)
from vllm.v1.serial_utils import run_method as run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase as WorkerWrapperBase

logger: Incomplete
PG_WAIT_TIMEOUT: int

class RayWorkerWrapper(WorkerWrapperBase):
    compiled_dag_cuda_device_set: bool
    def __init__(self, *args, **kwargs) -> None: ...
    rpc_rank: Incomplete
    def adjust_rank(self, rank_mapping: dict[int, int]) -> None: ...
    def execute_method(self, method: str | bytes, *args, **kwargs): ...
    def get_node_ip(self) -> str: ...
    def get_node_and_gpu_ids(self) -> tuple[str, list[int]]: ...
    def setup_device_if_necessary(self) -> None: ...
    def execute_model_ray(
        self,
        execute_model_input: tuple["SchedulerOutput", "GrammarOutput"]
        | tuple["SchedulerOutput", "GrammarOutput", "IntermediateTensors"],
    ) -> (
        ModelRunnerOutput
        | tuple["SchedulerOutput", "GrammarOutput", "IntermediateTensors"]
    ): ...
    def override_env_vars(self, vars: dict[str, str]): ...

ray_import_err: Incomplete

class FutureWrapper(Future):
    ref_or_refs: Incomplete
    aggregator: Incomplete
    def __init__(
        self, ref_or_refs, aggregator: KVOutputAggregator | None = None
    ) -> None: ...
    def result(self, timeout=None): ...

def ray_is_available() -> bool: ...
def assert_ray_available() -> None: ...
def initialize_ray_cluster(
    parallel_config: ParallelConfig, ray_address: str | None = None
): ...
def get_num_tpu_nodes() -> int: ...
def get_num_nodes_in_placement_group() -> int: ...
