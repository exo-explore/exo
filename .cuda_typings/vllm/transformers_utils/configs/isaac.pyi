from _typeshed import Incomplete
from transformers import Qwen3Config
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig

__all__ = ["IsaacConfig", "PixelShuffleSiglip2VisionConfig"]

class PixelShuffleSiglip2VisionConfig(Siglip2VisionConfig):
    model_type: str
    base_config_key: str
    pixel_shuffle_scale_factor: Incomplete
    num_patches: Incomplete
    def __init__(
        self, pixel_shuffle_scale_factor: int = 1, num_patches: int = 256, **kwargs
    ) -> None: ...

class IsaacConfig(Qwen3Config):
    model_type: str
    sub_configs: Incomplete
    text_config: Incomplete
    video_patch_size: Incomplete
    vision_max_num_patches: Incomplete
    vision_min_num_patches: Incomplete
    pixel_shuffle_scale: Incomplete
    max_sequence_length: Incomplete
    vision_token: Incomplete
    vision_config: Incomplete
    vision_attn_implementation: Incomplete
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        vision_patch_size: int = 16,
        vision_max_num_patches: int = 256,
        vision_min_num_patches: int | None = None,
        pixel_shuffle_scale: int = 1,
        max_sequence_length: int = 16384,
        vision_token: str = "<image>",
        vision_attn_implementation: str | None = None,
        **kwargs,
    ) -> None: ...
