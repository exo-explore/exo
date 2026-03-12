import contextlib
import zmq
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process
from multiprocessing.process import BaseProcess
from ray.util.placement_group import PlacementGroup as PlacementGroup
from vllm import envs as envs
from vllm.config import (
    CacheConfig as CacheConfig,
    ParallelConfig as ParallelConfig,
    VllmConfig as VllmConfig,
)
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.ray.ray_env import get_env_vars_to_copy as get_env_vars_to_copy
from vllm.utils.network_utils import (
    get_open_zmq_ipc_path as get_open_zmq_ipc_path,
    zmq_socket_ctx as zmq_socket_ctx,
)
from vllm.utils.system_utils import get_mp_context as get_mp_context
from vllm.v1.engine.coordinator import DPCoordinator as DPCoordinator
from vllm.v1.executor import Executor as Executor
from vllm.v1.utils import (
    get_engine_client_zmq_addr as get_engine_client_zmq_addr,
    shutdown as shutdown,
)

logger: Incomplete
STARTUP_POLL_PERIOD_MS: int

class CoreEngineState(Enum):
    NEW = ...
    CONNECTED = ...
    READY = ...

class CoreEngine:
    local: Incomplete
    identity: Incomplete
    state: Incomplete
    def __init__(self, index: int = 0, local: bool = True) -> None: ...

@dataclass
class EngineZmqAddresses:
    inputs: list[str]
    outputs: list[str]
    coordinator_input: str | None = ...
    coordinator_output: str | None = ...
    frontend_stats_publish_address: str | None = ...

@dataclass
class EngineHandshakeMetadata:
    addresses: EngineZmqAddresses
    parallel_config: dict[str, int | str | list[int]]

class CoreEngineProcManager:
    processes: list[BaseProcess]
    def __init__(
        self,
        target_fn: Callable,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ) -> None: ...
    def close(self) -> None: ...
    def join_first(self) -> None: ...
    def sentinels(self) -> list: ...
    def finished_procs(self) -> dict[str, int]: ...

@contextlib.contextmanager
def set_device_control_env_var(
    vllm_config: VllmConfig, local_dp_rank: int
) -> Iterator[None]: ...
def get_device_indices(
    device_control_env_var: str,
    local_dp_rank: int,
    world_size: int,
    local_world_size: int | None = None,
): ...

class CoreEngineActorManager:
    local_engine_actors: list[ray.ActorHandle]
    remote_engine_actors: list[ray.ActorHandle]
    env_vars_dict: Incomplete
    addresses: Incomplete
    executor_class: Incomplete
    log_stats: Incomplete
    created_placement_groups: Incomplete
    placement_group_is_local: Incomplete
    run_refs: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        placement_groups: list["PlacementGroup"] | None = None,
        local_dp_ranks: list[int] | None = None,
    ) -> None: ...
    @staticmethod
    def create_dp_placement_groups(
        vllm_config: VllmConfig,
    ) -> tuple[list["PlacementGroup"], list[int]]: ...
    @staticmethod
    def add_dp_placement_groups(
        old_vllm_config: VllmConfig, new_data_parallel_size: int
    ) -> tuple[list["PlacementGroup"], list[int]]: ...
    def scale_up_elastic_ep(
        self, cur_vllm_config: VllmConfig, new_data_parallel_size: int
    ) -> None: ...
    def scale_down_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None: ...
    def get_run_refs(self): ...
    def close(self) -> None: ...

def get_engine_zmq_addresses(
    vllm_config: VllmConfig, num_api_servers: int = 1
) -> EngineZmqAddresses: ...
@contextlib.contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
    num_api_servers: int = 1,
) -> Iterator[
    tuple[
        CoreEngineProcManager | CoreEngineActorManager | None,
        DPCoordinator | None,
        EngineZmqAddresses,
    ]
]: ...
def wait_for_engine_startup(
    handshake_socket: zmq.Socket,
    addresses: EngineZmqAddresses,
    core_engines: list[CoreEngine],
    parallel_config: ParallelConfig,
    coordinated_dp: bool,
    cache_config: CacheConfig,
    proc_manager: CoreEngineProcManager | None,
    coord_process: Process | None,
): ...
