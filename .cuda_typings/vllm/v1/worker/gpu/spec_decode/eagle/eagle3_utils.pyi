import torch.nn as nn
from _typeshed import Incomplete
from vllm.config import SpeculativeConfig as SpeculativeConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.models.interfaces import (
    SupportsEagle3 as SupportsEagle3,
    supports_eagle3 as supports_eagle3,
)

logger: Incomplete

def set_eagle3_aux_hidden_state_layers(
    model: nn.Module, spec_config: SpeculativeConfig
) -> None: ...
def get_eagle3_aux_layers_from_config(
    spec_config: SpeculativeConfig,
) -> tuple[int, ...] | None: ...
