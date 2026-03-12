import torch
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.sequence import IntermediateTensors as IntermediateTensors

class LegacyMixin:
    hf_to_vllm_mapper: Incomplete
    is_roberta: Incomplete
    padding_idx: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
