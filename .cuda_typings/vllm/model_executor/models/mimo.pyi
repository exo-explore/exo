import abc
import torch
import torch.nn as nn
from .utils import (
    PPMissingLayer as PPMissingLayer,
    is_pp_missing_parameter as is_pp_missing_parameter,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import get_pp_group as get_pp_group
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen2 import (
    Qwen2ForCausalLM as Qwen2ForCausalLM,
    Qwen2Model as Qwen2Model,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

logger: Incomplete

class MiMoModel(Qwen2Model):
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class MiMoForCausalLM(Qwen2ForCausalLM, nn.Module, metaclass=abc.ABCMeta):
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
