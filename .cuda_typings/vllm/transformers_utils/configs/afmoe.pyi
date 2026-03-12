from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig

__all__ = ["AfmoeConfig"]

class AfmoeConfig(PretrainedConfig):
    model_type: str
    vocab_size: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_dense_layers: Incomplete
    num_attention_heads: Incomplete
    num_key_value_heads: Incomplete
    head_dim: Incomplete
    hidden_act: Incomplete
    max_position_embeddings: Incomplete
    initializer_range: Incomplete
    rms_norm_eps: Incomplete
    use_cache: Incomplete
    rope_parameters: Incomplete
    rope_scaling: Incomplete
    moe_intermediate_size: Incomplete
    num_experts: Incomplete
    num_experts_per_tok: Incomplete
    num_shared_experts: Incomplete
    num_expert_groups: Incomplete
    num_limited_groups: Incomplete
    score_func: Incomplete
    route_norm: Incomplete
    route_scale: Incomplete
    global_attn_every_n_layers: Incomplete
    sliding_window: Incomplete
    layer_types: Incomplete
    attention_dropout: Incomplete
    mup_enabled: Incomplete
    n_group: Incomplete
    topk_group: Incomplete
    def __init__(
        self,
        vocab_size: int = 200192,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        moe_intermediate_size: int = 1408,
        num_hidden_layers: int = 32,
        num_dense_layers: int = 1,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-05,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: dict | None = None,
        rope_scaling: dict | None = None,
        num_experts: int = 64,
        num_experts_per_tok: int = 6,
        num_shared_experts: int = 2,
        num_expert_groups: int = 1,
        num_limited_groups: int = 1,
        score_func: str = "sigmoid",
        route_norm: bool = True,
        route_scale: float = 1.0,
        global_attn_every_n_layers: int = 4,
        sliding_window: int = 2048,
        layer_types: list[str] | None = None,
        attention_dropout: float = 0.0,
        mup_enabled: bool = False,
        n_group: int = 1,
        topk_group: int = 1,
        **kwargs,
    ) -> None: ...
