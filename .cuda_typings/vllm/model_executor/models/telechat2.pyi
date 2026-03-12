import abc
import torch
from .llama import LlamaDecoderLayer as LlamaDecoderLayer
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    WeightsMapper as WeightsMapper,
    is_pp_missing_parameter as is_pp_missing_parameter,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.llama import (
    LlamaForCausalLM as LlamaForCausalLM,
    LlamaModel as LlamaModel,
)

class TeleChat2Model(LlamaModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class TeleChat2ForCausalLM(LlamaForCausalLM, metaclass=abc.ABCMeta):
    hf_to_vllm_mapper: Incomplete
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
