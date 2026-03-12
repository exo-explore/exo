from _typeshed import Incomplete
from transformers import PretrainedConfig
from typing import Any

class AXK1Config(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    vocab_size: Incomplete
    max_position_embeddings: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    moe_intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_nextn_predict_layers: Incomplete
    num_attention_heads: Incomplete
    n_shared_experts: Incomplete
    n_routed_experts: Incomplete
    ep_size: Incomplete
    routed_scaling_factor: Incomplete
    kv_lora_rank: Incomplete
    q_lora_rank: Incomplete
    qk_rope_head_dim: Incomplete
    v_head_dim: Incomplete
    qk_nope_head_dim: Incomplete
    topk_method: Incomplete
    n_group: Incomplete
    topk_group: Incomplete
    num_experts_per_tok: Incomplete
    moe_layer_freq: Incomplete
    first_k_dense_replace: Incomplete
    norm_topk_prob: Incomplete
    scoring_func: Incomplete
    aux_loss_alpha: Incomplete
    seq_aux: Incomplete
    num_key_value_heads: Incomplete
    hidden_act: Incomplete
    initializer_range: Incomplete
    rms_norm_eps: Incomplete
    pretraining_tp: Incomplete
    use_cache: Incomplete
    rope_theta: Incomplete
    rope_scaling: Incomplete
    rope_parameters: Incomplete
    attention_bias: Incomplete
    attention_dropout: Incomplete
    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 61,
        num_nextn_predict_layers: int | None = 1,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 192,
        ep_size: int | None = 8,
        routed_scaling_factor: float | None = 2.5,
        kv_lora_rank: int | None = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 128,
        qk_nope_head_dim: int | None = 128,
        topk_method: str | None = "noaux_tc",
        n_group: int | None = 8,
        topk_group: int | None = 4,
        num_experts_per_tok: int | None = 8,
        moe_layer_freq: int | None = 1,
        first_k_dense_replace: int = 1,
        norm_topk_prob: bool = True,
        scoring_func: str | None = "sigmoid",
        aux_loss_alpha: float | None = 0.0001,
        seq_aux: float | None = True,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 131072,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float = 1e-06,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 163691,
        eos_token_id: int | None = 163691,
        pretraining_tp: int | None = 1,
        tie_word_embeddings: bool | None = False,
        rope_theta: float | None = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        **kwargs,
    ) -> None: ...
