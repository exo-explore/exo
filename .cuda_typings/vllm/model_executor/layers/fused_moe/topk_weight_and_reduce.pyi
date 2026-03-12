import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete

class TopKWeightAndReduceDelegate(mk.TopKWeightAndReduce):
    def __eq__(self, other): ...
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor: ...

class TopKWeightAndReduceNoOP(mk.TopKWeightAndReduce):
    def __eq__(self, other): ...
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor: ...

class TopKWeightAndReduceContiguous(mk.TopKWeightAndReduce):
    def __eq__(self, other): ...
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor: ...

class TopKWeightAndReduceNaiveBatched(mk.TopKWeightAndReduce):
    rank: Incomplete
    def __init__(self, rank: int) -> None: ...
    def __eq__(self, other): ...
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor: ...
