import multiprocessing
from _typeshed import Incomplete
from vllm.config import ParallelConfig as ParallelConfig
from vllm.logger import init_logger as init_logger
from vllm.utils.network_utils import make_zmq_socket as make_zmq_socket
from vllm.utils.system_utils import (
    get_mp_context as get_mp_context,
    set_process_title as set_process_title,
)
from vllm.v1.engine import (
    EngineCoreOutputs as EngineCoreOutputs,
    EngineCoreRequestType as EngineCoreRequestType,
)
from vllm.v1.serial_utils import MsgpackDecoder as MsgpackDecoder
from vllm.v1.utils import (
    get_engine_client_zmq_addr as get_engine_client_zmq_addr,
    shutdown as shutdown,
)

logger: Incomplete

class DPCoordinator:
    proc: multiprocessing.Process
    stats_publish_address: Incomplete
    coord_in_address: Incomplete
    coord_out_address: Incomplete
    def __init__(
        self, parallel_config: ParallelConfig, enable_wave_coordination: bool = True
    ) -> None: ...
    def get_stats_publish_address(self) -> str: ...
    def get_engine_socket_addresses(self) -> tuple[str, str]: ...
    def close(self) -> None: ...

class EngineState:
    request_counts: Incomplete
    def __init__(self) -> None: ...

class DPCoordinatorProc:
    ctx: Incomplete
    engines: Incomplete
    stats_update_interval_ms: Incomplete
    enable_wave_coordination: Incomplete
    def __init__(
        self,
        engine_count: int,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ) -> None: ...
    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ): ...
    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
    ): ...
