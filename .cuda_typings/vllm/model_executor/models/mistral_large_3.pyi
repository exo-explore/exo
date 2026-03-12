import abc
import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV3ForCausalLM as DeepseekV3ForCausalLM,
)

class MistralLarge3ForCausalLM(DeepseekV3ForCausalLM, metaclass=abc.ABCMeta):
    remapping: Incomplete
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
