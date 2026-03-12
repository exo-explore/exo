from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig
from typing import Any

__all__ = ["Lfm2MoeConfig"]

class Lfm2MoeConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    vocab_size: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    rope_parameters: Incomplete
    max_position_embeddings: Incomplete
    use_cache: Incomplete
    norm_eps: Incomplete
    num_attention_heads: Incomplete
    num_key_value_heads: Incomplete
    conv_bias: Incomplete
    conv_L_cache: Incomplete
    num_dense_layers: Incomplete
    moe_intermediate_size: Incomplete
    num_experts_per_tok: Incomplete
    num_experts: Incomplete
    use_expert_bias: Incomplete
    routed_scaling_factor: Incomplete
    norm_topk_prob: Incomplete
    layer_types: Incomplete
    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2048,
        intermediate_size: int = 7168,
        moe_intermediate_size: int = 1792,
        num_hidden_layers: int = 32,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 128000,
        use_cache: bool = True,
        norm_eps: float = 1e-05,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        conv_bias: bool = False,
        conv_L_cache: int = 3,
        num_dense_layers: int = 2,
        num_experts_per_tok: int = 4,
        num_experts: int = 32,
        use_expert_bias: bool = True,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        layer_types: list[str] | None = None,
        **kwargs,
    ) -> None: ...
