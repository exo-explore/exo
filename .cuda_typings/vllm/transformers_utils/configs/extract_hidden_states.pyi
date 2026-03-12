import os
from transformers import PretrainedConfig
from vllm.transformers_utils.utils import (
    without_trust_remote_code as without_trust_remote_code,
)

class ExtractHiddenStatesConfig(PretrainedConfig):
    model_type: str
    def __init__(
        self,
        model: PretrainedConfig | dict | None = None,
        method: str | None = "extract_hidden_states",
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> ExtractHiddenStatesConfig: ...
    def to_json_string(self, use_diff: bool = True) -> str: ...
