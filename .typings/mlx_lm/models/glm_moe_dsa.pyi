"""Type stubs for mlx_lm.models.glm_moe_dsa"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import BaseModelArgs
from .deepseek_v32 import Model as DSV32Model

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    moe_layer_freq: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: Dict[str, Any]
    attention_bias: bool
    rope_scaling: Dict[str, Any] | None
    rope_theta: float | None

class Model(DSV32Model):
    def __init__(self, config: ModelArgs) -> None: ...
