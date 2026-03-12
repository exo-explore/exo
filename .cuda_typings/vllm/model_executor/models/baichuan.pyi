import abc
import torch
from .interfaces import (
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
)
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
from transformers import PretrainedConfig as PretrainedConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
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
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    row_parallel_weight_loader as row_parallel_weight_loader,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

class BaiChuanMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x): ...

class BaiChuanAttention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    position_embedding: Incomplete
    max_position_embeddings: Incomplete
    W_pack: Incomplete
    o_proj: Incomplete
    attn: Incomplete
    rotary_emb: Incomplete
    scaling: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        position_embedding: str,
        rope_parameters: dict,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class BaiChuanDecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        position_embedding: str,
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

class BaiChuanModel(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        position_embedding: str = "ROPE",
    ) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class BaiChuanBaseForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsQuant, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    config: Incomplete
    tp_size: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        position_embedding: str = "ROPE",
    ) -> None: ...
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
    def lm_head_weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ): ...

class BaichuanForCausalLM(BaiChuanBaseForCausalLM, metaclass=abc.ABCMeta):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class BaiChuanForCausalLM(BaiChuanBaseForCausalLM, metaclass=abc.ABCMeta):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
