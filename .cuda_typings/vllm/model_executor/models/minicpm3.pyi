import abc
import torch
from .utils import make_layers as make_layers
from _typeshed import Incomplete
from torch import nn
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.models.minicpm import (
    MiniCPMDecoderLayer as MiniCPMDecoderLayer,
    MiniCPMForCausalLM as MiniCPMForCausalLM,
    MiniCPMModel as MiniCPMModel,
)

class MiniCPM3Attention(nn.Module):
    hidden_size: Incomplete
    qk_nope_head_dim: Incomplete
    qk_rope_head_dim: Incomplete
    qk_head_dim: Incomplete
    v_head_dim: Incomplete
    q_lora_rank: Incomplete
    kv_lora_rank: Incomplete
    num_heads: Incomplete
    num_local_heads: Incomplete
    scaling: Incomplete
    max_position_embeddings: Incomplete
    q_a_proj: Incomplete
    q_a_layernorm: Incomplete
    q_b_proj: Incomplete
    kv_a_proj_with_mqa: Incomplete
    kv_a_layernorm: Incomplete
    kv_b_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class MiniCPM3DecoderLayer(MiniCPMDecoderLayer): ...
class MiniCPM3Model(MiniCPMModel): ...

class MiniCPM3ForCausalLM(MiniCPMForCausalLM, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
