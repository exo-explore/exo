from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig
from typing import Any

logger: Incomplete
VIT_TIMM_DIM_BY_NAME: dict[str, tuple[int, int, int, int]]
OPENAI_CLIP_MEAN: Incomplete
OPENAI_CLIP_STD: Incomplete

class RadioConfig(PretrainedConfig):
    model_type: str
    model_name: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    qkv_bias: Incomplete
    qk_normalization: Incomplete
    norm_type: Incomplete
    layer_norm_eps: Incomplete
    initializer_factor: Incomplete
    hidden_act: Incomplete
    cpe_max_size: Incomplete
    norm_mean: Incomplete
    norm_std: Incomplete
    register_multiple: Incomplete
    teachers: Incomplete
    cls_token_per_teacher: Incomplete
    def __init__(
        self,
        model_name: str,
        image_size: int = 224,
        patch_size: int = 16,
        qkv_bias: bool = True,
        qk_normalization: bool = False,
        norm_type: str = "layer_norm",
        layer_norm_eps: float = 1e-06,
        initializer_factor: float = 1.0,
        hidden_act: str = "gelu",
        cpe_max_size: int = 2048,
        norm_mean: tuple[float, float, float] | list = ...,
        norm_std: tuple[float, float, float] | list = ...,
        register_multiple: int | None = None,
        teachers: list[dict[str, Any]] | None = None,
        cls_token_per_teacher: bool = False,
        **kwargs,
    ) -> None: ...
