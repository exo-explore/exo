import os
from _typeshed import Incomplete
from transformers import DeepseekV2Config as DeepseekV2Config, PretrainedConfig
from vllm.transformers_utils.utils import (
    without_trust_remote_code as without_trust_remote_code,
)

class EAGLEConfig(PretrainedConfig):
    model_type: str
    model: Incomplete
    truncated_vocab_size: Incomplete
    def __init__(
        self,
        model: PretrainedConfig | dict | None = None,
        truncated_vocab_size: int | None = None,
        method: str | None = "eagle",
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> EAGLEConfig: ...
    def to_json_string(self, use_diff: bool = True) -> str: ...
