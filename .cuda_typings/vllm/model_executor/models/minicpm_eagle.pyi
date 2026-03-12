import abc
import torch
from .interfaces import (
    SupportsEagle as SupportsEagle,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    maybe_prefix as maybe_prefix,
    process_eagle_weight as process_eagle_weight,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import PretrainedConfig as PretrainedConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

class EagleMiniCPMDecoderLayer(nn.Module):
    config: Incomplete
    cache_config: Incomplete
    quant_config: Incomplete
    hidden_size: Incomplete
    max_position_embeddings: Incomplete
    prefix: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class EagleMiniCPMModel(nn.Module):
    config: Incomplete
    cache_config: Incomplete
    quant_config: Incomplete
    vocab_size: Incomplete
    fc: Incomplete
    input_norm1: Incomplete
    input_norm2: Incomplete
    embed_tokens: Incomplete
    num_experts: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", start_layer: int = 0
    ) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class EagleMiniCPMForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    embedding_modules: Incomplete
    prefix: Incomplete
    vllm_config: Incomplete
    config: Incomplete
    cache_config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    scale_width: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
