from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig
from typing import Any

class Step3p5Config(PretrainedConfig):
    model_type: str
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_attention_heads: Incomplete
    num_attention_groups: Incomplete
    num_hidden_layers: Incomplete
    max_seq_len: Incomplete
    vocab_size: Incomplete
    rms_norm_eps: Incomplete
    use_moe: Incomplete
    moe_intermediate_size: Incomplete
    moe_every_n_layer: Incomplete
    moe_num_experts: Incomplete
    num_experts_per_tok: Incomplete
    moe_top_k: Incomplete
    moe_layer_offset: Incomplete
    rope_theta: Incomplete
    rope_scaling: Incomplete
    head_dim: Incomplete
    share_expert_dim: Incomplete
    norm_expert_weight: Incomplete
    max_position_embeddings: Incomplete
    moe_router_activation: Incomplete
    moe_router_scaling_factor: Incomplete
    use_moe_router_bias: Incomplete
    need_fp32_gate: Incomplete
    att_impl_type: Incomplete
    use_head_wise_attn_gate: Incomplete
    layer_types: Incomplete
    use_rope_layers: Incomplete
    yarn_only_types: Incomplete
    attention_other_setting: Incomplete
    num_nextn_predict_layers: Incomplete
    swiglu_limits: Incomplete
    swiglu_limits_shared: Incomplete
    bos_token_id: Incomplete
    eos_token_id: Incomplete
    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        rms_norm_eps: float = 1e-05,
        moe_every_n_layer: int = 2,
        use_moe: bool = False,
        moe_intermediate_size: int = 10240,
        moe_num_experts: int = 16,
        moe_top_k: int = 4,
        moe_layer_offset: int = 0,
        rope_theta: float | list[float] | None = 500000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        share_expert_dim: int | None = None,
        norm_expert_weight: bool = True,
        bos_token_id: list[int] | int | None = None,
        eos_token_id: list[int] | int | None = None,
        moe_router_activation: str = "softmax",
        moe_router_scaling_factor: float = 1.0,
        att_impl_type: str = "GQA",
        use_head_wise_attn_gate: bool = False,
        use_moe_router_bias: bool = True,
        need_fp32_gate: bool = True,
        layer_types: list[str] | None = None,
        use_rope_layers: list[bool] | None = None,
        yarn_only_types: list[str] | None = None,
        attention_other_setting: dict[str, Any] | None = None,
        num_nextn_predict_layers: int = 0,
        swiglu_limits: list[float] | None = None,
        swiglu_limits_shared: list[float] | None = None,
        max_position_embeddings: int | None = None,
        **kwargs,
    ) -> None: ...
