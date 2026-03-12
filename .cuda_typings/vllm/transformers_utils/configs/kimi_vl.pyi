from _typeshed import Incomplete
from transformers import DeepseekV2Config
from transformers.configuration_utils import PretrainedConfig
from vllm.transformers_utils.configs.moonvit import MoonViTConfig as MoonViTConfig

class KimiVLConfig(PretrainedConfig):
    model_type: str
    vision_config: Incomplete
    text_config: Incomplete
    ignore_index: Incomplete
    media_placeholder_token_id: Incomplete
    def __init__(
        self,
        vision_config: dict | MoonViTConfig | None = None,
        text_config: dict | DeepseekV2Config | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        **kwargs,
    ) -> None: ...
