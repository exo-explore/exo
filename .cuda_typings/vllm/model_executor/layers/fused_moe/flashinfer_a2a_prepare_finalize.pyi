import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from vllm.distributed import get_ep_group as get_ep_group
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase as All2AllManagerBase,
)
from vllm.forward_context import get_forward_context as get_forward_context
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input as moe_kernel_quantize_input,
)
from vllm.utils.flashinfer import (
    nvfp4_block_scale_interleave as nvfp4_block_scale_interleave,
)

def get_local_sizes(): ...

class FlashInferA2APrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    num_dispatchers_: Incomplete
    all2all_manager: Incomplete
    def __init__(self, num_dispatchers: int = 1) -> None: ...
    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat: ...
    def max_num_tokens_per_rank(self) -> int | None: ...
    def topk_indices_dtype(self) -> torch.dtype | None: ...
    def num_dispatchers(self) -> int: ...
    def output_is_reduced(self) -> bool: ...
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType: ...
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None: ...

def flashinfer_alltoall_dispatch(
    all2all_manager: All2AllManagerBase,
    global_num_tokens_cpu: list[int],
    x: torch.Tensor,
    gs: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    num_experts: int,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
): ...
def flashinfer_alltoall_combine(
    all2all_manager: All2AllManagerBase,
    output: torch.Tensor,
    top_k: int,
    token_count: int,
    alltoall_info,
): ...
