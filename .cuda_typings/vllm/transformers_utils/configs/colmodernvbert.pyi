from _typeshed import Incomplete
from transformers import PretrainedConfig

class ColModernVBertConfig(PretrainedConfig):
    model_type: str
    embedding_dim: Incomplete
    image_token_id: Incomplete
    pixel_shuffle_factor: Incomplete
    hidden_size: Incomplete
    text_config: Incomplete
    vision_config: Incomplete
    def __init__(
        self, embedding_dim: int = 128, vlm_config: dict | None = None, **kwargs
    ) -> None: ...
    @property
    def image_seq_len(self) -> int: ...
    def get_text_config(self, **kwargs): ...
