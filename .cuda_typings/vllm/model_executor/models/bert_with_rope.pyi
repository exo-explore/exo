import torch
from .bert import BertPooler as BertPooler
from .interfaces import (
    SupportsCrossEncoding as SupportsCrossEncoding,
    SupportsQuant as SupportsQuant,
)
from .interfaces_base import default_pooling_type as default_pooling_type
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import PretrainedConfig as PretrainedConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import (
    get_act_and_mul_fn as get_act_and_mul_fn,
    get_act_fn as get_act_fn,
)
from vllm.model_executor.layers.attention import (
    EncoderOnlyAttention as EncoderOnlyAttention,
)
from vllm.model_executor.layers.fused_moe import (
    activation_without_mul as activation_without_mul,
    fused_topk as fused_topk,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler as DispatchPooler
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors

class BertWithRopeEmbedding(nn.Module):
    word_embeddings: Incomplete
    token_type_embeddings: Incomplete
    LayerNorm: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class BertWithRopeAttention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    head_dim: Incomplete
    num_kv_heads: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    qkv_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    out_proj: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        bias: bool = True,
        rotary_kwargs: dict | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class BertWithRopeGatedMLP(nn.Module):
    act_fn: Incomplete
    gate_up_proj: Incomplete
    down_proj: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class BertWithRopeMLP(nn.Module):
    act_fn: Incomplete
    up_proj: Incomplete
    down_proj: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class NomicMoE(nn.Module):
    tp_size: Incomplete
    num_total_experts: Incomplete
    top_k: Incomplete
    hidden_size: Incomplete
    total_intermediate_size: Incomplete
    intermediate_size: Incomplete
    hidden_act: Incomplete
    params_dtype: Incomplete
    router: Incomplete
    w1: Incomplete
    w2: Incomplete
    bias: Incomplete
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        params_dtype: torch.dtype | None = None,
        tp_size: int | None = None,
    ) -> None: ...
    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str
    ): ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class BertWithRopeBlock(nn.Module):
    attn: Incomplete
    mlp: Incomplete
    attn_ln: Incomplete
    mlp_ln: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        moe: bool = False,
        bias: bool = True,
        rotary_kwargs: dict | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor): ...

class BertWithRopeEncoder(nn.Module):
    layers: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        bias: bool = True,
        rotary_kwargs: dict | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class BertWithRope(nn.Module, SupportsQuant):
    hf_to_vllm_mapper: Incomplete
    vllm_config: Incomplete
    add_pooling_layer: Incomplete
    config: Incomplete
    embeddings: Incomplete
    encoder: Incomplete
    pooler: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        add_pooling_layer: bool = False,
    ) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class NomicBertModel(BertWithRope):
    hf_to_vllm_mapper: Incomplete

class GteNewModel(BertWithRope):
    hf_to_vllm_mapper: Incomplete
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs
    ) -> None: ...
    def split_up_gate_proj(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
    def ignore_unnecessary_layers(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ): ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class SnowflakeGteNewModel(GteNewModel):
    hf_to_vllm_mapper: Incomplete

class JinaRobertaModel(BertWithRope):
    hf_to_vllm_mapper: Incomplete
    def jina_merge_lora_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class GteNewForSequenceClassification(nn.Module, SupportsCrossEncoding):
    is_pooling_model: bool
    new: Incomplete
    classifier: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
