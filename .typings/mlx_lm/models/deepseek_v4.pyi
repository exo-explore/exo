"""Type stubs for mlx_lm.models.deepseek_v4"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import ArraysCache, RotatingKVCache
from .switch_layers import SwitchGLU

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: int
    num_experts_per_tok: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    sliding_window: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    compress_ratios: Optional[List[int]]
    compress_rope_theta: float
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]
    rms_norm_eps: float
    swiglu_limit: float
    attention_bias: bool
    max_position_embeddings: int

class DeepseekV4RoPE(nn.Module):
    dims: int
    freqs: mx.array

    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict[str, Any]] = None,
    ) -> None: ...
    def __call__(
        self,
        x: mx.array,
        offset: int = 0,
        inverse: bool = False,
    ) -> mx.array: ...

class HyperConnection(nn.Module):
    dim: int
    hc_mult: int
    norm_eps: float

    def __init__(
        self,
        dim: int,
        hc_mult: int,
        norm_eps: float,
        sinkhorn_iters: int,
        hc_eps: float,
    ) -> None: ...

class HyperHead(nn.Module):
    dim: int
    hc_mult: int

    def __init__(
        self,
        dim: int,
        hc_mult: int,
        norm_eps: float,
        hc_eps: float,
    ) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Compressor(nn.Module):
    dim: int
    head_dim: int
    rope_head_dim: int
    compress_ratio: int
    overlap: bool
    wkv_gate: nn.Linear
    ape: mx.array
    norm: nn.RMSNorm
    rope: DeepseekV4RoPE

    def __init__(
        self,
        dim: int,
        compress_ratio: int,
        head_dim: int,
        rope_head_dim: int,
        rms_norm_eps: float,
        rope: DeepseekV4RoPE,
    ) -> None: ...
    def __call__(
        self,
        x: mx.array,
        cache: "DeepseekV4Cache",
        offset: Any,
        key: str = ...,
    ) -> mx.array: ...

class Indexer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        compress_ratio: int,
        rope: DeepseekV4RoPE,
    ) -> None: ...

class _CompressorBranch:
    buffer_kv: Optional[mx.array]
    buffer_gate: Optional[mx.array]
    prev_kv: Optional[mx.array]
    prev_gate: Optional[mx.array]
    pool: Optional[mx.array]
    buffer_lengths: Optional[List[int]]
    pool_lengths: Optional[List[int]]
    buffer_count: int
    _new_pool_lengths: Optional[List[int]]

    def __init__(self) -> None: ...

class DeepseekV4Cache:
    local: RotatingKVCache
    offset: int
    keys: Optional[mx.array]
    values: Optional[mx.array]
    state: Any
    meta_state: Any
    nbytes: int
    _branches: Dict[str, _CompressorBranch]
    _pending_lengths: Optional[List[int]]

    def __init__(self, sliding_window: int) -> None: ...
    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]: ...
    def is_trimmable(self) -> bool: ...
    def trim(self, n: int) -> int: ...
    def empty(self) -> bool: ...
    def size(self) -> int: ...
    def prepare(
        self,
        *,
        left_padding: Optional[List[int]] = None,
        lengths: Optional[List[int]] = None,
        right_padding: Optional[List[int]] = None,
    ) -> None: ...
    def finalize(self) -> None: ...
    def filter(self, batch_indices: mx.array) -> None: ...
    def extend(self, other: "DeepseekV4Cache") -> None: ...
    def extract(self, idx: int) -> "DeepseekV4Cache": ...
    @classmethod
    def merge(cls, caches: List["DeepseekV4Cache"]) -> "DeepseekV4Cache": ...

class V4Attention(nn.Module):
    args: ModelArgs
    layer_id: int
    dim: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    nope_head_dim: int
    n_groups: int
    q_lora_rank: int
    o_lora_rank: int
    window: int
    eps: float
    scale: float
    compress_ratio: int
    wqkv_a: nn.Linear
    q_norm: nn.RMSNorm
    wq_b: nn.Linear
    kv_norm: nn.RMSNorm
    attn_sink: mx.array
    wo_a: nn.Linear
    wo_b: nn.Linear
    rope: DeepseekV4RoPE
    compressor: Compressor
    indexer: Indexer

    def __init__(self, args: ModelArgs, layer_id: int) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class DeepseekV4MLP(nn.Module):
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float = 0.0,
    ) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class MoEGate(nn.Module):
    weight: mx.array

    def __init__(self, args: ModelArgs, layer_id: int) -> None: ...
    def __call__(
        self, x: mx.array, input_ids: mx.array
    ) -> tuple[mx.array, mx.array]: ...

class DeepseekV4MoE(nn.Module):
    num_experts_per_tok: int
    switch_mlp: SwitchGLU
    gate: MoEGate
    shared_experts: DeepseekV4MLP

    def __init__(self, args: ModelArgs, layer_id: int) -> None: ...
    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array: ...

class DeepseekV4Block(nn.Module):
    attn_norm: nn.RMSNorm
    attn: V4Attention
    hc_attn: HyperConnection
    ffn_norm: nn.RMSNorm
    ffn: DeepseekV4MoE
    hc_ffn: HyperConnection

    def __init__(self, args: ModelArgs, layer_id: int) -> None: ...
    def __call__(
        self,
        h: mx.array,
        cache: Optional[Any],
        input_ids: mx.array,
    ) -> mx.array: ...

class DeepseekV4Model(nn.Module):
    args: ModelArgs
    vocab_size: int
    embed_tokens: nn.Embedding
    layers: list[DeepseekV4Block]
    norm: nn.RMSNorm
    hc_head: HyperHead

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array: ...

class Model(nn.Module):
    args: ModelArgs
    model_type: str
    model: DeepseekV4Model
    lm_head: nn.Linear

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array: ...
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
    def make_cache(self) -> list[RotatingKVCache | DeepseekV4Cache]: ...
    @property
    def layers(self) -> list[DeepseekV4Block]: ...
