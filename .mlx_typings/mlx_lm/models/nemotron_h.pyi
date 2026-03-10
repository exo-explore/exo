from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .cache import ArraysCache, KVCache
from .switch_layers import SwitchMLP


@dataclass
class ModelArgs:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    mamba_num_heads: int
    mamba_head_dim: int
    mamba_proj_bias: bool
    ssm_state_size: int
    conv_kernel: int
    n_groups: int
    mlp_bias: bool
    layer_norm_epsilon: float
    use_bias: bool
    use_conv_bias: bool
    hybrid_override_pattern: List[str]
    head_dim: Optional[int]
    moe_intermediate_size: Optional[int]
    moe_shared_expert_intermediate_size: Optional[int]
    n_group: Optional[int]
    n_routed_experts: Optional[int]
    n_shared_experts: Optional[int]
    topk_group: Optional[int]
    num_experts_per_tok: Optional[int]
    norm_topk_prob: Optional[bool]
    routed_scaling_factor: Optional[float]
    time_step_limit: Optional[Tuple[float, float]]
    time_step_min: Optional[float]
    time_step_max: Optional[float]

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs: ...
    def __post_init__(self) -> None: ...


class NemotronHMamba2Mixer(nn.Module):
    num_heads: int
    hidden_size: int
    ssm_state_size: int
    conv_kernel_size: int
    intermediate_size: int
    n_groups: int
    head_dim: int
    conv_dim: int
    conv1d: nn.Conv1d
    in_proj: nn.Linear
    dt_bias: mx.array
    A_log: mx.array
    D: mx.array
    out_proj: nn.Linear

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array],
        cache: Optional[ArraysCache] = None,
    ) -> mx.array: ...


class NemotronHAttention(nn.Module):
    hidden_size: int
    num_heads: int
    head_dim: int
    num_key_value_heads: int
    scale: float
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array: ...


class NemotronHMLP(nn.Module):
    up_proj: nn.Linear
    down_proj: nn.Linear

    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...


class NemotronHMoE(nn.Module):
    num_experts_per_tok: int
    switch_mlp: SwitchMLP
    shared_experts: NemotronHMLP

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...


class NemotronHBlock(nn.Module):
    block_type: str
    norm: nn.RMSNorm
    mixer: NemotronHMamba2Mixer | NemotronHAttention | NemotronHMLP | NemotronHMoE

    def __init__(self, args: ModelArgs, block_type: str) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...


class NemotronHModel(nn.Module):
    embeddings: nn.Embedding
    layers: list[NemotronHBlock]
    norm_f: nn.RMSNorm
    fa_idx: int
    ssm_idx: int

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...


class Model(nn.Module):
    args: ModelArgs
    backbone: NemotronHModel
    lm_head: nn.Linear
    model_type: str

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...
    @property
    def layers(self) -> list[NemotronHBlock]: ...
    def make_cache(self) -> list[ArraysCache | KVCache]: ...
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
