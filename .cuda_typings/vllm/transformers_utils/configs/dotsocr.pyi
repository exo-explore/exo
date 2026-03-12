from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2 import Qwen2Config
from typing import Any

class DotsVisionConfig(PretrainedConfig):
    model_type: str
    embed_dim: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    num_channels: Incomplete
    patch_size: Incomplete
    spatial_merge_size: Incomplete
    temporal_patch_size: Incomplete
    rms_norm_eps: Incomplete
    use_bias: Incomplete
    attn_implementation: Incomplete
    initializer_range: Incomplete
    init_merger_std: Incomplete
    is_causal: Incomplete
    post_norm: Incomplete
    gradient_checkpointing: Incomplete
    def __init__(
        self,
        embed_dim: int = 1536,
        hidden_size: int = 1536,
        intermediate_size: int = 4224,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-05,
        use_bias: bool = False,
        attn_implementation: str = "flash_attention_2",
        initializer_range: float = 0.02,
        init_merger_std: float = 0.02,
        is_causal: bool = False,
        post_norm: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs: Any,
    ) -> None: ...

class DotsOCRConfig(Qwen2Config):
    model_type: str
    image_token_id: Incomplete
    video_token_id: Incomplete
    vision_config: Incomplete
    def __init__(
        self,
        image_token_id: int = 151665,
        video_token_id: int = 151656,
        vision_config: dict | None = None,
        *args,
        **kwargs,
    ) -> None: ...
    def save_pretrained(self, save_directory, **kwargs) -> None: ...
