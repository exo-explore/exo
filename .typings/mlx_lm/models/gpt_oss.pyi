"""Type stubs for mlx_lm.models.gpt_oss"""

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import KVCache
from .switch_layers import SwitchGLU

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_local_experts: int
    num_experts_per_tok: int
    vocab_size: int
    rms_norm_eps: float
    sliding_window: int
    layer_types: Optional[List[str]]

def mlx_topk(a: mx.array, k: int, axis: int = -1) -> tuple[mx.array, mx.array]: ...

class AttentionBlock(nn.Module):
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    num_key_value_groups: int
    sinks: mx.array
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    sm_scale: float
    rope: nn.Module

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class TransformerBlock(nn.Module):
    self_attn: AttentionBlock
    mlp: MLPBlock

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class MLPBlock(nn.Module):
    hidden_size: int
    num_local_experts: int
    num_experts_per_tok: int
    experts: SwitchGLU
    router: nn.Linear
    sharding_group: Optional[mx.distributed.Group]

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class GptOssMoeModel(nn.Module):
    embed_tokens: nn.Embedding
    norm: nn.RMSNorm
    layer_types: List[str]
    layers: list[TransformerBlock]
    window_size: int
    swa_idx: int
    ga_idx: int

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class Model(nn.Module):
    model_type: str
    model: GptOssMoeModel
    lm_head: nn.Linear

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...
    @property
    def layers(self) -> list[nn.Module]: ...
    def make_cache(self) -> list[KVCache]: ...
