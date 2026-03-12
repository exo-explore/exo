import abc
import torch
from .interfaces import SupportsLoRA as SupportsLoRA, SupportsPP as SupportsPP
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import PhiConfig as PhiConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

class PhiAttention(nn.Module):
    hidden_size: Incomplete
    head_size: Incomplete
    num_heads: Incomplete
    qkv_proj: Incomplete
    dense: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: PhiConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, position_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class PhiMLP(nn.Module):
    fc1: Incomplete
    fc2: Incomplete
    act: Incomplete
    def __init__(
        self,
        config: PhiConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states): ...

class PhiLayer(nn.Module):
    input_layernorm: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        config: PhiConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, position_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class PhiModel(nn.Module):
    config: Incomplete
    quant_config: Incomplete
    embed_tokens: Incomplete
    final_layernorm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class PhiForCausalLM(nn.Module, SupportsLoRA, SupportsPP, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
