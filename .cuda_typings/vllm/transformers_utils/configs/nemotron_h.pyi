from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig

logger: Incomplete

class NemotronHConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    vocab_size: Incomplete
    tie_word_embeddings: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    hybrid_override_pattern: Incomplete
    mtp_hybrid_override_pattern: Incomplete
    num_attention_heads: Incomplete
    head_dim: Incomplete
    sliding_window: Incomplete
    max_position_embeddings: Incomplete
    attention_dropout: Incomplete
    hidden_dropout: Incomplete
    num_key_value_heads: Incomplete
    mlp_hidden_act: Incomplete
    attention_bias: Incomplete
    mlp_bias: Incomplete
    use_bias: Incomplete
    initializer_range: Incomplete
    layer_norm_epsilon: Incomplete
    residual_in_fp32: Incomplete
    use_cache: Incomplete
    num_logits_to_keep: Incomplete
    use_mamba_kernels: Incomplete
    n_groups: Incomplete
    mamba_head_dim: Incomplete
    ssm_state_size: Incomplete
    mamba_num_heads: Incomplete
    conv_kernel: Incomplete
    expand: Incomplete
    mamba_hidden_act: Incomplete
    time_step_min: Incomplete
    time_step_max: Incomplete
    time_step_limit: Incomplete
    time_step_floor: Incomplete
    use_conv_bias: Incomplete
    mamba_proj_bias: Incomplete
    chunk_size: Incomplete
    rescale_prenorm_residual: Incomplete
    n_routed_experts: Incomplete
    n_shared_experts: Incomplete
    moe_intermediate_size: Incomplete
    moe_shared_expert_intermediate_size: Incomplete
    moe_latent_size: Incomplete
    num_experts_per_tok: Incomplete
    routed_scaling_factor: Incomplete
    n_group: Incomplete
    topk_group: Incomplete
    norm_topk_prob: Incomplete
    def __init__(
        self,
        vocab_size: int = 131072,
        tie_word_embeddings: bool = False,
        hidden_size: int = 4096,
        intermediate_size: int = 21504,
        num_hidden_layers: int = 52,
        hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
        mtp_hybrid_override_pattern: str = "*E",
        num_attention_heads: int = 32,
        head_dim: int = 128,
        num_key_value_heads: int = 8,
        mlp_hidden_act: str = "relu2",
        attention_bias: bool = False,
        mlp_bias: bool = False,
        use_bias: bool = False,
        initializer_range: float = 0.02,
        layer_norm_epsilon: float = 1e-05,
        residual_in_fp32: bool = False,
        use_cache: bool = True,
        num_logits_to_keep: int = 1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        sliding_window=None,
        max_position_embeddings: int = 4096,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        use_mamba_kernels: bool = True,
        ssm_state_size: int = 128,
        mamba_num_heads: int = 128,
        mamba_n_groups: int = 8,
        mamba_head_dim: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_hidden_act: str = "silu",
        mamba_dt_min: float = 0.001,
        mamba_dt_max: float = 0.1,
        mamba_dt_limit=...,
        mamba_dt_init_floor: float = 0.0001,
        mamba_conv_bias: bool = True,
        mamba_proj_bias: bool = False,
        mamba_chunk_size: int = 256,
        rescale_prenorm_residual: bool = True,
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        moe_intermediate_size: int = 7688,
        moe_shared_expert_intermediate_size: int = 7688,
        moe_latent_size=None,
        num_experts_per_tok: int = 2,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        **kwargs,
    ) -> None: ...
    @property
    def layers_block_type(self): ...
