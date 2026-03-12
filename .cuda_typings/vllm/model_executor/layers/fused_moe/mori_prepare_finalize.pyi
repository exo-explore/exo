import mori
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.platforms import current_platform as current_platform

logger: Incomplete

class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    mori_op: Incomplete
    num_dispatchers_: Incomplete
    max_tokens_per_rank: Incomplete
    use_fp8_dispatch: Incomplete
    def __init__(
        self,
        mori_op: mori.ops.EpDispatchCombineOp,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
    ) -> None: ...
    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat: ...
    def output_is_reduced(self) -> bool: ...
    def num_dispatchers(self): ...
    def max_num_tokens_per_rank(self) -> int | None: ...
    def topk_indices_dtype(self) -> torch.dtype | None: ...
    def supports_async(self) -> bool: ...
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
