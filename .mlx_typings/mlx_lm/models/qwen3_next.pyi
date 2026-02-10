"""Type stubs for mlx_lm.models.qwen3_next"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .switch_layers import SwitchGLU

class Qwen3NextMLP(nn.Module):
    gate_proj: nn.Linear
    down_proj: nn.Linear
    up_proj: nn.Linear

    def __init__(self, dim: int, hidden_dim: int) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Qwen3NextGatedDeltaNet(nn.Module):
    hidden_size: int
    num_v_heads: int
    num_k_heads: int
    head_k_dim: int
    head_v_dim: int
    key_dim: int
    value_dim: int
    conv_kernel_size: int
    conv_dim: int
    conv1d: nn.Conv1d
    in_proj_qkvz: nn.Linear
    in_proj_ba: nn.Linear
    dt_bias: mx.array
    A_log: mx.array
    out_proj: nn.Linear

    def __init__(self, config: Any) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class Qwen3NextAttention(nn.Module):
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    scale: float
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear

    def __init__(self, args: Any) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class Qwen3NextSparseMoeBlock(nn.Module):
    norm_topk_prob: bool
    num_experts: int
    top_k: int
    gate: nn.Linear
    switch_mlp: SwitchGLU
    shared_expert: Qwen3NextMLP
    shared_expert_gate: nn.Linear

    def __init__(self, args: Any) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Qwen3NextDecoderLayer(nn.Module):
    is_linear: bool
    linear_attn: Qwen3NextGatedDeltaNet
    self_attn: Qwen3NextAttention
    input_layernorm: nn.RMSNorm
    post_attention_layernorm: nn.RMSNorm
    mlp: Qwen3NextMLP | Qwen3NextSparseMoeBlock

    def __init__(self, args: Any, layer_idx: int) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class Qwen3NextModel(nn.Module):
    embed_tokens: nn.Embedding
    layers: list[Qwen3NextDecoderLayer]
    norm: nn.RMSNorm

    def __init__(self, args: Any) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class Model(nn.Module):
    model_type: str
    model: Qwen3NextModel
    lm_head: nn.Linear

    def __init__(self, args: Any) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
    @property
    def layers(self) -> list[Qwen3NextDecoderLayer]: ...
