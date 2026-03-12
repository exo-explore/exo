import os
from .config_parser_base import ConfigParserBase as ConfigParserBase
from .gguf_utils import (
    check_gguf_file as check_gguf_file,
    is_gguf as is_gguf,
    is_remote_gguf as is_remote_gguf,
    split_remote_gguf as split_remote_gguf,
)
from .repo_utils import (
    file_or_path_exists as file_or_path_exists,
    get_hf_file_to_dict as get_hf_file_to_dict,
    list_repo_files as list_repo_files,
    try_get_local_file as try_get_local_file,
    with_retry as with_retry,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from functools import cache
from pathlib import Path
from transformers import GenerationConfig, PretrainedConfig
from typing import Any
from vllm import envs as envs
from vllm.logger import init_logger as init_logger
from vllm.transformers_utils.repo_utils import (
    is_mistral_model_repo as is_mistral_model_repo,
)
from vllm.transformers_utils.utils import (
    parse_safetensors_file_metadata as parse_safetensors_file_metadata,
    without_trust_remote_code as without_trust_remote_code,
)

MISTRAL_CONFIG_NAME: str
logger: Incomplete

class LazyConfigDict(dict):
    def __getitem__(self, key): ...

def is_rope_parameters_nested(rope_parameters: dict[str, Any]) -> bool: ...

class HFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]: ...

class MistralConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]: ...

ConfigFormat: Incomplete

def get_config_parser(config_format: str) -> ConfigParserBase: ...
def register_config_parser(config_format: str): ...
def set_default_rope_theta(config: PretrainedConfig, default_theta: float) -> None: ...
def patch_rope_parameters(config: PretrainedConfig) -> None: ...
def patch_rope_parameters_dict(rope_parameters: dict[str, Any]) -> None: ...
def uses_mrope(config: PretrainedConfig) -> bool: ...
def thinker_uses_mrope(config: PretrainedConfig) -> bool: ...
def uses_xdrope_dim(config: PretrainedConfig) -> int: ...
def is_encoder_decoder(config: PretrainedConfig) -> bool: ...
def is_interleaved(config: PretrainedConfig) -> bool: ...
def maybe_override_with_speculators(
    model: str,
    tokenizer: str | None,
    trust_remote_code: bool,
    revision: str | None = None,
    vllm_speculative_config: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[str, str | None, dict[str, Any] | None]: ...
def get_config(
    model: str | Path,
    trust_remote_code: bool,
    revision: str | None = None,
    code_revision: str | None = None,
    config_format: str | ConfigFormat = "auto",
    hf_overrides_kw: dict[str, Any] | None = None,
    hf_overrides_fn: Callable[[PretrainedConfig], PretrainedConfig] | None = None,
    **kwargs,
) -> PretrainedConfig: ...
@cache
def get_pooling_config(
    model: str, revision: str | None = "main"
) -> dict[str, Any] | None: ...
def parse_pooling_type(pooling_name: str): ...
@cache
def get_sentence_transformer_tokenizer_config(
    model: str | Path, revision: str | None = "main"
) -> dict[str, Any] | None: ...
def maybe_register_config_serialize_by_value() -> None: ...
def get_hf_image_processor_config(
    model: str | Path,
    hf_token: bool | str | None = None,
    revision: str | None = None,
    **kwargs,
) -> dict[str, Any]: ...
def get_hf_text_config(config: PretrainedConfig): ...
def try_get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    config_format: str | ConfigFormat = "auto",
) -> GenerationConfig | None: ...
def try_get_safetensors_metadata(model: str, *, revision: str | None = None): ...
def try_get_tokenizer_config(
    pretrained_model_name_or_path: str | os.PathLike,
    trust_remote_code: bool,
    revision: str | None = None,
) -> dict[str, Any] | None: ...
@cache
def try_get_dense_modules(
    model: str | Path, revision: str | None = None
) -> list[dict[str, Any]] | None: ...
def get_safetensors_params_metadata(
    model: str, *, revision: str | None = None
) -> dict[str, Any]: ...
