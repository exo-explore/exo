import torch
from .granitemoe import (
    GraniteMoeAttention as GraniteMoeAttention,
    GraniteMoeMoE as GraniteMoeMoE,
    GraniteMoeModel as GraniteMoeModel,
)
from .interfaces import SupportsLoRA as SupportsLoRA, SupportsPP as SupportsPP
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers.models.granitemoeshared import (
    GraniteMoeSharedConfig as GraniteMoeSharedConfig,
)
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import get_pp_group as get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
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
from vllm.sequence import IntermediateTensors as IntermediateTensors

class GraniteMoeSharedMLP(nn.Module):
    input_size: Incomplete
    hidden_size: Incomplete
    input_linear: Incomplete
    output_linear: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        config: GraniteMoeSharedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteMoeSharedDecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    block_sparse_moe: Incomplete
    shared_mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    residual_multiplier: Incomplete
    def __init__(
        self,
        config: GraniteMoeSharedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class GraniteMoeSharedModel(nn.Module):
    config: Incomplete
    quant_config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    embedding_multiplier: Incomplete
    norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class GraniteMoeSharedForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    fall_back_to_pt_during_load: bool
    packed_modules_mapping: Incomplete
    embedding_modules: Incomplete
    config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
