import abc
import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import PretrainedConfig
from typing import Any
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_pp_group as get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
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
    DEFAULT_VOCAB_PADDING_SIZE as DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    LoaderFunction as LoaderFunction,
    composed_weight_loader as composed_weight_loader,
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    extract_layer_index as extract_layer_index,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.sequence import IntermediateTensors as IntermediateTensors

class Plamo3Config(PretrainedConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    num_attention_heads: int
    head_dim: int
    num_key_value_heads: int
    interleaved_sliding_window: list[int | None]
    sliding_window_pattern: int
    rope_parameters: dict[str, Any]
    rope_local_theta: int
    intermediate_size: int
    vocab_size: int

def rms_norm_weight_loader(offset: float) -> LoaderFunction: ...

class DenseMLP(nn.Module):
    hidden_size: Incomplete
    intermediate_size: Incomplete
    gate_up_proj: Incomplete
    act: Incomplete
    down_proj: Incomplete
    def __init__(
        self,
        config: Plamo3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Plamo3AttentionMixer(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    attn: Incomplete
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor: ...

class Plamo3DecoderLayer(nn.Module):
    mixer: Incomplete
    mlp: Incomplete
    pre_mixer_norm: Incomplete
    post_mixer_norm: Incomplete
    pre_mlp_norm: Incomplete
    post_mlp_norm: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, prefix: str = "", **kwargs: Any
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Plamo3Decoder(torch.nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Plamo3Model(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    org_vocab_size: Incomplete
    embed_tokens: Incomplete
    make_empty_intermediate_tensors: Incomplete
    layers: Incomplete
    norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class Plamo3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    config: Incomplete
    vllm_config: Incomplete
    model_config: Incomplete
    scheduler_config: Incomplete
    model: Incomplete
    vocab_size: Incomplete
    unpadded_vocab_size: Incomplete
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
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
