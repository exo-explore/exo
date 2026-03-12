from _typeshed import Incomplete
from vllm.config import ParallelConfig as ParallelConfig
from vllm.logger import init_logger as init_logger

logger: Incomplete

def override_envs_for_eplb(parallel_config: ParallelConfig) -> None: ...
