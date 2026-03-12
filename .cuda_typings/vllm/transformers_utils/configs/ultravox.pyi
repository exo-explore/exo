import transformers
from _typeshed import Incomplete
from typing import Any

class UltravoxConfig(transformers.PretrainedConfig):
    wrapped_model_config: transformers.PretrainedConfig
    model_type: str
    audio_token: str
    is_composition: bool
    ignore_index: Incomplete
    audio_token_index: Incomplete
    hidden_size: Incomplete
    stack_factor: Incomplete
    norm_init: Incomplete
    projector_act: Incomplete
    projector_ln_mid: Incomplete
    num_projector_layers: Incomplete
    text_model_id: Incomplete
    audio_model_id: Incomplete
    audio_config: Incomplete
    def __init__(
        self,
        audio_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | None = None,
        audio_model_id: str | None = None,
        text_model_id: str | None = None,
        ignore_index: int = -100,
        audio_token_index: int = 32000,
        hidden_size: int = 4096,
        stack_factor: int = 8,
        norm_init: float = 0.4,
        projector_act: str = "swiglu",
        projector_ln_mid: bool = False,
        num_projector_layers: int = 0,
        **kwargs,
    ) -> None: ...
    def __setattr__(self, key, value): ...
    @property
    def text_config(self) -> transformers.PretrainedConfig: ...
