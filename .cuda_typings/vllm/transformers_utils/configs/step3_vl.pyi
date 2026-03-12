from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig
from typing import Any

class Step3VisionEncoderConfig(PretrainedConfig):
    model_type: str
    hidden_size: Incomplete
    intermediate_size: Incomplete
    output_hidden_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    num_channels: Incomplete
    patch_size: Incomplete
    image_size: Incomplete
    layer_norm_eps: Incomplete
    hidden_act: Incomplete
    def __init__(
        self,
        hidden_size: int = 1792,
        intermediate_size: int = 3072,
        output_hidden_size: int = 4096,
        num_hidden_layers: int = 63,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 728,
        patch_size: int = 14,
        hidden_act: str = "quick_gelu",
        layer_norm_eps: float = 1e-05,
        **kwargs,
    ) -> None: ...

class Step3TextConfig(PretrainedConfig):
    model_type: str
    architectures: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_attention_heads: Incomplete
    num_attention_groups: Incomplete
    num_hidden_layers: Incomplete
    max_seq_len: Incomplete
    vocab_size: Incomplete
    rms_norm_eps: Incomplete
    moe_intermediate_size: Incomplete
    moe_num_experts: Incomplete
    moe_top_k: Incomplete
    rope_parameters: Incomplete
    max_position_embedding: Incomplete
    share_expert_dim: Incomplete
    share_q_dim: Incomplete
    head_dim: Incomplete
    norm_expert_weight: Incomplete
    moe_layers_enum: Incomplete
    def __init__(
        self,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        num_attention_heads: int = 64,
        num_attention_groups: int = 1,
        num_hidden_layers: int = 61,
        max_seq_len: int = 65536,
        vocab_size: int = 128815,
        rms_norm_eps: float = 1e-05,
        moe_intermediate_size: int = 5120,
        moe_num_experts: int = 48,
        moe_top_k: int = 3,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embedding: int = 65536,
        share_expert_dim: int = 5120,
        share_q_dim: int = 2048,
        head_dim: int = 256,
        norm_expert_weight: bool = False,
        moe_layers_enum: tuple[int, ...] = ...,
        **kwargs,
    ) -> None: ...

class Step3VLConfig(PretrainedConfig):
    model_type: str
    vision_config: Incomplete
    text_config: Incomplete
    understand_projector_stride: Incomplete
    projector_bias: Incomplete
    hidden_size: Incomplete
    image_token_id: Incomplete
    def __init__(
        self,
        vision_config: dict | Step3VisionEncoderConfig | None = None,
        text_config: dict | Step3TextConfig | None = None,
        understand_projector_stride: int = 1,
        projector_bias: bool = True,
        image_token_id: int = 128001,
        **kwargs,
    ) -> None: ...
