import os
from transformers import PretrainedConfig
from typing import Any

__all__ = ["SpeculatorsConfig"]

class SpeculatorsConfig(PretrainedConfig):
    model_type: str
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> SpeculatorsConfig: ...
    @classmethod
    def extract_transformers_pre_trained_config(
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]: ...
    @classmethod
    def extract_vllm_speculative_config(
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]: ...
    @classmethod
    def validate_speculators_config(cls, config_dict: dict[str, Any]) -> None: ...
    @classmethod
    def build_vllm_speculative_config(
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]: ...
