import abc
import torch
import torch.nn as nn
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.models.llama import (
    LlamaDecoderLayer as LlamaDecoderLayer,
    LlamaForCausalLM as LlamaForCausalLM,
    LlamaModel as LlamaModel,
)

class TeleFLMModel(LlamaModel):
    use_mup: Incomplete
    input_mult: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = ...,
    ) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...

class TeleFLMForCausalLM(LlamaForCausalLM, metaclass=abc.ABCMeta):
    use_mup: Incomplete
    mup_scale_factor: Incomplete
    output_mult: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
