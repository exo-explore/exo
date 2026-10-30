"""Type stubs for the GLM-5.3-Flash model (glm5_next.py next to this file)."""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import SwitchGLU

LINEAR_ATTENTION: str
SPARSE_ATTENTION: str

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    layer_types: list[str]
    mlp_layer_types: list[str]
    indexer_types: list[str]
    linear_attn_config: dict[str, Any]
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    topk_method: str
    scoring_func: str
    swiglu_limit: float
    first_k_dense_replace: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    attention_bias: bool
    mla_use_nope: bool
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_compress: bool
    index_kpool_always_select_tail: bool
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    tie_word_embeddings: bool

class HyperConnection(nn.Module):
    hc_mult: int
    fn: mx.array
    base: mx.array
    scale: mx.array
    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]: ...

class MLP(nn.Module):
    swiglu_limit: float
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear
    def __init__(self, args: ModelArgs, intermediate_size: int) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class MoEGate(nn.Module):
    weight: mx.array
    e_score_correction_bias: mx.array
    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x: mx.array
    ) -> tuple[mx.array, mx.array]: ...

class MoE(nn.Module):
    gate: MoEGate
    switch_mlp: SwitchGLU
    shared_experts: MLP
    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class GatedRMSNorm(nn.Module):
    eps: float
    weight: mx.array
    def __init__(self, dims: int, eps: float) -> None: ...
    def __call__(self, x: mx.array, gate: mx.array) -> mx.array: ...

class KimiDeltaAttention(nn.Module):
    num_heads: int
    head_dim: int
    conv_kernel: int
    lower_bound: float | None
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    q_conv: nn.Conv1d
    k_conv: nn.Conv1d
    v_conv: nn.Conv1d
    f_a_proj: nn.Linear
    f_b_proj: nn.Linear
    dt_bias: mx.array
    A_log: mx.array
    b_proj: nn.Linear
    g_a_proj: nn.Linear
    g_b_proj: nn.Linear
    o_norm: GatedRMSNorm
    o_proj: nn.Linear
    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(self, x: mx.array, cache: ArraysCache | None = None) -> mx.array: ...

class Indexer(nn.Module):
    n_heads: int
    head_dim: int
    index_topk: int
    kpool: int
    wq_b: nn.Linear
    wk: nn.Linear
    k_norm: nn.LayerNorm
    weights_proj: nn.Linear
    index_kpool_compress_ape: mx.array
    index_kpool_compress_gate: mx.array
    softmax_scale: float
    def __init__(self, args: ModelArgs) -> None: ...
    @property
    def dense_limit(self) -> int: ...
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x: mx.array, qr: mx.array, cache: KVCache | BatchKVCache | None = None
    ) -> mx.array | None: ...

class Glm5NextAttention(nn.Module):
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    v_head_dim: int
    scale: float
    q_a_proj: nn.Linear
    q_a_layernorm: nn.RMSNorm
    q_b_proj: nn.Linear
    kv_a_proj_with_mqa: nn.Linear
    kv_a_layernorm: nn.RMSNorm
    embed_q: MultiLinear
    unembed_out: MultiLinear
    o_proj: nn.Linear
    indexer: Indexer
    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(self, x: mx.array, cache: CacheList | None = None) -> mx.array: ...

class DecoderLayer(nn.Module):
    is_linear: bool
    self_attn: KimiDeltaAttention | Glm5NextAttention
    mlp: MoE | MLP
    input_layernorm: nn.RMSNorm
    post_attention_layernorm: nn.RMSNorm
    attn_hc: HyperConnection
    ffn_hc: HyperConnection
    def __init__(self, args: ModelArgs, layer_idx: int) -> None: ...
    def __call__(
        self, x: mx.array, cache: ArraysCache | CacheList | None = None
    ) -> mx.array: ...

class Glm5NextModel(nn.Module):
    hc_mult: int
    embed_tokens: nn.Embedding
    layers: list[DecoderLayer]
    norm: nn.RMSNorm
    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: list[ArraysCache | CacheList] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...

class Model(nn.Module):
    args: ModelArgs
    model_type: str
    model: Glm5NextModel
    lm_head: nn.Linear
    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: list[ArraysCache | CacheList] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...
    @property
    def layers(self) -> list[DecoderLayer]: ...
    def make_cache(self) -> list[ArraysCache | CacheList]: ...
    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]: ...
