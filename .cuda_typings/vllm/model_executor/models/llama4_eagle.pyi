import abc
import torch
import torch.nn as nn
from .interfaces import SupportsMultiModal as SupportsMultiModal
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    maybe_prefix as maybe_prefix,
    process_eagle_weight as process_eagle_weight,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.quantization.torchao import (
    TorchAOConfig as TorchAOConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.llama4 import (
    Llama4DecoderLayer as Llama4DecoderLayer,
    Llama4ForCausalLM as Llama4ForCausalLM,
)
from vllm.model_executor.models.utils import extract_layer_index as extract_layer_index

logger: Incomplete

class LlamaModel(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    fc: Incomplete
    norm: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
    ) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def validate_and_update_config(
        self, start_layer_id: int, quant_config: QuantizationConfig | None = None
    ) -> None: ...

class EagleLlama4ForCausalLM(Llama4ForCausalLM, metaclass=abc.ABCMeta):
    config: Incomplete
    model: Incomplete
    logits_processor: Incomplete
    lm_head: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def get_language_model(self) -> torch.nn.Module: ...
    embed_input_ids: Incomplete
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None: ...
