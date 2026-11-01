"""Type stubs for the vendored DeepSeek V4 model (model.py next to this file)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    intermediate_size: int = 0
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: Optional[int] = 1
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    compress_ratios: Optional[List[int]] = None
    compress_rope_theta: float = 160000.0
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    rms_norm_eps: float = 1e-6
    swiglu_limit: float = 10.0
    attention_bias: bool = False
    max_position_embeddings: int = 1048576
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_hash_layers: int = 3
    num_nextn_predict_layers: int = 1
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5

# exo fixes (see the header of model.py)
_SDPA_MAX_SCORES: int

def _pow2_ceil(x: mx.array) -> mx.array: ...
def _round_e2m1(v: mx.array) -> mx.array: ...
def _fp8_kv_sim(kv: mx.array, rd: int) -> mx.array: ...
def _fp4_indexer_sim(x: mx.array) -> mx.array: ...

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
        offset: int | mx.array,
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
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
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
