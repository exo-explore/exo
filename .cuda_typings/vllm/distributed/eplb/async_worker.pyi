import threading
import torch
from .eplb_state import EplbModelState as EplbModelState, EplbState as EplbState
from .rebalance_execute import transfer_layer as transfer_layer
from _typeshed import Incomplete
from torch.distributed import ProcessGroup as ProcessGroup
from vllm.distributed.parallel_state import get_eplb_group as get_eplb_group
from vllm.logger import init_logger as init_logger

logger: Incomplete

def start_async_worker(
    state: EplbState, is_profile: bool = False
) -> threading.Thread: ...
def run_rebalance_experts(
    model_state: EplbModelState,
    eplb_state: EplbState,
    physical_to_logical_map_cpu: torch.Tensor,
) -> None: ...
async def transfer_run_periodically(
    state: EplbState,
    eplb_group: ProcessGroup,
    cuda_stream: torch.cuda.Stream,
    is_profile: bool = False,
) -> None: ...
