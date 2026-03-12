from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig

__all__ = ["Qwen3NextConfig"]

class Qwen3NextConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    base_model_tp_plan: Incomplete
    base_model_pp_plan: Incomplete
    vocab_size: Incomplete
    max_position_embeddings: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    num_key_value_heads: Incomplete
    hidden_act: Incomplete
    initializer_range: Incomplete
    rms_norm_eps: Incomplete
    use_cache: Incomplete
    rope_parameters: Incomplete
    partial_rotary_factor: Incomplete
    attention_bias: Incomplete
    attention_dropout: Incomplete
    head_dim: Incomplete
    layer_types: Incomplete
    linear_conv_kernel_dim: Incomplete
    linear_key_head_dim: Incomplete
    linear_value_head_dim: Incomplete
    linear_num_key_heads: Incomplete
    linear_num_value_heads: Incomplete
    decoder_sparse_step: Incomplete
    moe_intermediate_size: Incomplete
    shared_expert_intermediate_size: Incomplete
    num_experts_per_tok: Incomplete
    num_experts: Incomplete
    norm_topk_prob: Incomplete
    output_router_logits: Incomplete
    router_aux_loss_coef: Incomplete
    mlp_only_layers: Incomplete
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-06,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters=None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        head_dim: int = 256,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        num_experts_per_tok: int = 10,
        num_experts: int = 512,
        norm_topk_prob: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers=None,
        layer_types=None,
        **kwargs,
    ) -> None: ...
