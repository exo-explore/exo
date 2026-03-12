import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from vllm.compilation.counter import compilation_counter as compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphWrapper as CUDAGraphWrapper
from vllm.compilation.wrapper import reset_compile_wrapper as reset_compile_wrapper
from vllm.config import (
    CompilationMode as CompilationMode,
    set_current_vllm_config as set_current_vllm_config,
)
from vllm.distributed import (
    get_dp_group as get_dp_group,
    get_ep_group as get_ep_group,
    get_pcp_group as get_pcp_group,
    get_tp_group as get_tp_group,
)
from vllm.distributed.elastic_ep.standby_state import (
    create_standby_groups as create_standby_groups,
    get_standby_dp_group as get_standby_dp_group,
    get_standby_ep_group as get_standby_ep_group,
    pop_standby_groups as pop_standby_groups,
)
from vllm.distributed.parallel_state import (
    prepare_communication_buffer_for_model as prepare_communication_buffer_for_model,
)
from vllm.distributed.stateless_coordinator import (
    StatelessGroupCoordinator as StatelessGroupCoordinator,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoEParallelConfig as FusedMoEParallelConfig,
)
from vllm.v1.engine import (
    ReconfigureDistributedRequest as ReconfigureDistributedRequest,
    ReconfigureRankType as ReconfigureRankType,
)
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper as UBatchWrapper
from vllm.v1.worker.workspace import (
    lock_workspace as lock_workspace,
    unlock_workspace as unlock_workspace,
)

logger: Incomplete

def batch_transfer_weights(
    model: nn.Module,
    is_sender: bool,
    peer_rank: int,
    dp_group: StatelessGroupCoordinator,
    expert_weights: Sequence[Iterable[torch.Tensor]],
) -> None: ...
def broadcast_expert_mapping(
    physical_to_logical: torch.Tensor | None,
    num_local_physical_experts: int | None,
    num_logical_experts: int | None,
    dp_group: StatelessGroupCoordinator,
    device: torch.device,
    src_rank: int = 0,
) -> tuple[torch.Tensor, int, int]: ...

class ElasticEPScalingExecutor:
    worker_ref: Incomplete
    reconfig_request: Incomplete
    def __init__(self, worker) -> None: ...
    @property
    def worker(self): ...
    def execute(self, execute_method: str, *args, **kwargs): ...
    def create_standby_groups(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None: ...
    def transfer_weights(self, old_dp_size: int, new_dp_size: int) -> None: ...
    def broadcast_expert_mapping(self) -> None: ...
    def switch_and_remove(self) -> None: ...
    def switch_and_prepare(self) -> None: ...
    def perform_eplb_reshuffle(self, new_dp_size: int | None = None) -> None: ...
    def receive_weights(self) -> None: ...
    def receive_expert_mapping(self) -> tuple[torch.Tensor, int, int]: ...
    def prepare_new_worker(self) -> None: ...
