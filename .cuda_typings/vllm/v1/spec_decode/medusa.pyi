import torch
import torch.nn as nn
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.forward_context import set_forward_context as set_forward_context
from vllm.logger import init_logger as init_logger
from vllm.model_executor.model_loader import get_model as get_model
from vllm.model_executor.models.interfaces import (
    is_mixture_of_experts as is_mixture_of_experts,
)
from vllm.v1.sample.metadata import SamplingMetadata as SamplingMetadata

logger: Incomplete

class MedusaProposer:
    vllm_config: Incomplete
    spec_config: Incomplete
    device: Incomplete
    max_num_tokens: Incomplete
    hidden_size: Incomplete
    dtype: Incomplete
    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None: ...
    def propose(
        self,
        target_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor: ...
    model: Incomplete
    def load_model(self, target_model: nn.Module) -> None: ...
    def dummy_run(self, num_tokens: int) -> None: ...
