import threading
import torch
import types
from _typeshed import Incomplete
from vllm import forward_context as forward_context
from vllm.forward_context import ForwardContext as ForwardContext
from vllm.logger import init_logger as init_logger
from vllm.utils.torch_utils import current_stream as current_stream

logger: Incomplete

class UBatchContext:
    id: Incomplete
    comm_stream: Incomplete
    compute_stream: Incomplete
    forward_context: Incomplete
    ready_barrier: Incomplete
    cpu_wait_event: Incomplete
    cpu_signal_event: Incomplete
    current_stream: Incomplete
    gpu_comm_done_event: Incomplete
    gpu_compute_done_event: Incomplete
    schedule: Incomplete
    recv_hook: Incomplete
    def __init__(
        self,
        id: int,
        comm_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        forward_context: ForwardContext,
        ready_barrier: threading.Barrier,
        cpu_wait_event: threading.Event,
        cpu_signal_event: threading.Event,
        gpu_comm_done_event: torch.Event,
        gpu_compute_done_event: torch.Event,
        schedule: str = "default",
    ) -> None: ...
    def __enter__(self): ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ): ...
    def update_stream(self, stream) -> None: ...
    def switch_to_comm(self) -> None: ...
    def switch_to_compute(self) -> None: ...
    def switch_to_comm_sync(self) -> None: ...
    def switch_to_compute_sync(self) -> None: ...
    def maybe_run_recv_hook(self) -> None: ...
    def yield_(self) -> None: ...
    def yield_and_switch_from_compute_to_comm(self) -> None: ...
    def yield_and_switch_from_comm_to_compute(self) -> None: ...

def dbo_enabled() -> bool: ...
def dbo_current_ubatch_id() -> int: ...

dbo_maybe_run_recv_hook: Incomplete
dbo_yield: Incomplete
dbo_yield_and_switch_from_compute_to_comm: Incomplete
dbo_yield_and_switch_from_comm_to_compute: Incomplete
dbo_switch_to_comm: Incomplete
dbo_switch_to_compute: Incomplete
dbo_switch_to_comm_sync: Incomplete
dbo_switch_to_compute_sync: Incomplete

def dbo_register_recv_hook(recv_hook) -> None: ...
def dbo_get_previous_event(func, *args, **kwargs): ...
def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
    schedule: str = "default",
) -> list[UBatchContext]: ...
