import abc
import torch
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch.nn import Parameter as Parameter
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import set_weight_attrs as set_weight_attrs
from vllm.model_executor.models.llama import LlamaForCausalLM as LlamaForCausalLM

class Fairseq2LlamaForCausalLM(LlamaForCausalLM, metaclass=abc.ABCMeta):
    tp_rank: Incomplete
    tp_size: Incomplete
    allow_patterns_overrides: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def flag_sharded_weights(self, params: dict[str, Parameter]): ...
    def reshape_fairseq2_weights(
        self, name: str, loaded_weight: torch.Tensor, params: dict[str, Parameter]
    ) -> tuple[str, torch.Tensor]: ...
