from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig
from vllm.logger import init_logger as init_logger

logger: Incomplete

class KimiLinearConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    vocab_size: Incomplete
    hidden_size: Incomplete
    head_dim: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    num_key_value_heads: Incomplete
    hidden_act: Incomplete
    initializer_range: Incomplete
    rms_norm_eps: Incomplete
    use_cache: Incomplete
    rope_parameters: Incomplete
    q_lora_rank: Incomplete
    kv_lora_rank: Incomplete
    qk_nope_head_dim: Incomplete
    qk_rope_head_dim: Incomplete
    v_head_dim: Incomplete
    mla_use_nope: Incomplete
    num_experts: Incomplete
    num_experts_per_token: Incomplete
    moe_renormalize: Incomplete
    num_shared_experts: Incomplete
    routed_scaling_factor: Incomplete
    moe_router_activation_func: Incomplete
    moe_intermediate_size: Incomplete
    first_k_dense_replace: Incomplete
    moe_layer_freq: Incomplete
    use_grouped_topk: Incomplete
    num_expert_group: Incomplete
    topk_group: Incomplete
    num_nextn_predict_layers: Incomplete
    linear_attn_config: Incomplete
    def __init__(
        self,
        model_type: str = "kimi_linear",
        vocab_size: int = 163840,
        hidden_size: int = 4096,
        head_dim=None,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads=None,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-06,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        rope_parameters=None,
        tie_word_embeddings: bool = False,
        moe_intermediate_size: int | None = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: int | None = None,
        num_experts_per_token: int | None = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        mla_use_nope: bool | None = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: dict | None = None,
        **kwargs,
    ) -> None: ...
    @property
    def is_mla(self): ...
    @property
    def is_moe(self): ...
    @property
    def is_linear_attn(self) -> bool: ...
    def is_kda_layer(self, layer_idx: int): ...
