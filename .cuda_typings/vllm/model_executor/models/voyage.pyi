from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.qwen3 import Qwen3Model as Qwen3Model
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper

WeightItem: Incomplete

class VoyageQwen3BidirectionalEmbedModel(Qwen3Model):
    hf_to_vllm_mapper: Incomplete
    linear: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, *args, **kwargs): ...
    def load_weights(self, weights: Iterable[WeightItem]) -> set[str]: ...
