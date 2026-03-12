from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig

__all__ = ["Qwen3_5MoeConfig", "Qwen3_5MoeTextConfig"]

class Qwen3_5MoeTextConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    base_model_tp_plan: Incomplete
    base_model_pp_plan: Incomplete
    base_config_key: str
    vocab_size: Incomplete
    max_position_embeddings: Incomplete
    hidden_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    num_key_value_heads: Incomplete
    hidden_act: Incomplete
    initializer_range: Incomplete
    rms_norm_eps: Incomplete
    use_cache: Incomplete
    attention_bias: Incomplete
    attention_dropout: Incomplete
    head_dim: Incomplete
    rope_parameters: Incomplete
    layer_types: Incomplete
    linear_conv_kernel_dim: Incomplete
    linear_key_head_dim: Incomplete
    linear_value_head_dim: Incomplete
    linear_num_key_heads: Incomplete
    linear_num_value_heads: Incomplete
    moe_intermediate_size: Incomplete
    shared_expert_intermediate_size: Incomplete
    num_experts_per_tok: Incomplete
    num_experts: Incomplete
    output_router_logits: Incomplete
    router_aux_loss_coef: Incomplete
    pad_token_id: Incomplete
    bos_token_id: Incomplete
    eos_token_id: Incomplete
    tie_word_embeddings: Incomplete
    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 2048,
        num_hidden_layers: int = 40,
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
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        num_experts_per_tok: int = 8,
        num_experts: int = 256,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        layer_types=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ) -> None: ...

class Qwen3_5MoeVisionConfig(PretrainedConfig):
    model_type: str
    base_config_key: str
    depth: Incomplete
    hidden_size: Incomplete
    hidden_act: Incomplete
    intermediate_size: Incomplete
    num_heads: Incomplete
    in_channels: Incomplete
    patch_size: Incomplete
    spatial_merge_size: Incomplete
    temporal_patch_size: Incomplete
    out_hidden_size: Incomplete
    num_position_embeddings: Incomplete
    initializer_range: Incomplete
    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None: ...

class Qwen3_5MoeConfig(PretrainedConfig):
    model_type: str
    sub_configs: Incomplete
    keys_to_ignore_at_inference: Incomplete
    vision_config: Incomplete
    text_config: Incomplete
    image_token_id: Incomplete
    video_token_id: Incomplete
    vision_start_token_id: Incomplete
    vision_end_token_id: Incomplete
    tie_word_embeddings: Incomplete
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None: ...
