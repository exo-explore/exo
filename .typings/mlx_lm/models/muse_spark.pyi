"""Type stubs for mlx_lm.models.muse_spark"""

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import KVCache, RotatingKVCache
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU

TOKENIZER_DEFAULTS: dict[str, Any]
POOLED_TP_MAX_TOKENS: int

@dataclass
class TextArgs(BaseModelArgs):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    post_norm_eps: float
    rope_parameters: dict[str, Any]
    layer_rope_theta: list[float]
    layer_types: list[str]
    sliding_window: int
    final_logit_softcapping: float
    output_multiplier: float
    num_local_experts: int
    num_experts_per_tok: int
    moe_hidden_size: int
    moe_intermediate_size: int
    qk_scale_factor: float = ...
    routed_scaling_factor: float = ...
    num_shared_experts: int = ...
    mlp_layer_types: list[str] | None = ...

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict[str, Any]
    vision_config: dict[str, Any] | None = ...
    image_token_id: int = ...
    out_hidden_size: int | None = ...
    projector_hidden_size: int | None = ...
    text: TextArgs = ...

class InputScaleRMSNorm(nn.Module):
    weight: mx.array
    eps: float
    def __init__(self, dims: int, eps: float) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

def residual_gate_coefficients(
    gate: mx.array, *, temperature: float = ..., eps: float = ...
) -> tuple[mx.array, mx.array]: ...
def norm_gain(weight: mx.array) -> mx.array: ...
def interleave_rope_rows(w: mx.array, head_dim: int) -> mx.array: ...
def shard_experts_inplace(
    module: nn.Module, sharding: str, group: mx.distributed.Group
) -> None: ...

class Attention(nn.Module):
    n_heads: int
    n_kv_heads: int
    head_dim: int
    scale: float
    query_scale: float
    eps: float
    qkvg_proj: nn.Linear
    o_proj: nn.Linear
    use_rope: bool
    rope: nn.RoPE
    def __init__(self, args: TextArgs, *, use_rope: bool) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array: ...

class MoEGate(nn.Module):
    weight: mx.array
    e_score_correction_bias: mx.array
    def __init__(self, dim: int, num_experts: int) -> None: ...

class MuseSparkExperts(SwitchGLU):
    post_expert_norm: InputScaleRMSNorm

class MoE(nn.Module):
    top_k: int
    gate: MoEGate
    pre_expert_proj: nn.Linear
    pre_expert_norm: nn.RMSNorm
    experts: MuseSparkExperts
    post_expert_proj: nn.Linear
    sharding_group: mx.distributed.Group | None
    def __init__(self, args: TextArgs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class TransformerBlock(nn.Module):
    use_sliding: bool
    post_norm_eps: float
    input_layernorm: nn.RMSNorm
    self_attn: Attention
    pre_feedforward_layernorm: nn.RMSNorm
    mlp: MoE
    post_feedforward_layernorm: InputScaleRMSNorm
    add_attn_alpha: mx.array
    add_attn_beta: mx.array
    add_ffn_alpha: mx.array
    add_ffn_beta: mx.array
    def __init__(
        self, args: TextArgs, *, use_rope: bool, use_sliding: bool
    ) -> None: ...
    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None = None,
        cache: Any | None = None,
        attn_input: mx.array | None = None,
        next_norm: nn.Module | None = None,
    ) -> tuple[mx.array, mx.array | None]: ...

class MuseSparkTextModel(PipelineMixin, nn.Module):
    args: TextArgs
    sliding_window: int
    embed_tokens: nn.Embedding
    layers: list[TransformerBlock]
    norm: nn.RMSNorm
    def __init__(self, args: TextArgs) -> None: ...
    @property
    def fa_idx(self) -> int: ...
    @property
    def swa_idx(self) -> int | None: ...
    def embed(self, inputs: mx.array) -> mx.array: ...
    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...

class MuseSparkModel(nn.Module):
    language_model: MuseSparkTextModel
    def __init__(self, args: ModelArgs) -> None: ...
    @property
    def layers(self) -> list[TransformerBlock]: ...

class Model(nn.Module):
    args: ModelArgs
    model_type: str
    model: MuseSparkModel
    lm_head: nn.Linear
    sharding_group: mx.distributed.Group | None
    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array: ...
    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]: ...
    def shard(self, group: mx.distributed.Group | None = None) -> None: ...
    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], bool]: ...
    @property
    def cast_predicate(self) -> Callable[[str], bool]: ...
    @property
    def layers(self) -> list[TransformerBlock]: ...
    def make_cache(self) -> list[KVCache | RotatingKVCache]: ...
