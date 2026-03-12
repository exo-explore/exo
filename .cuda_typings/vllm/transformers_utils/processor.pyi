from _typeshed import Incomplete
from functools import lru_cache
from transformers import processing_utils
from transformers.processing_utils import ProcessorMixin
from typing import Any
from vllm.config import ModelConfig as ModelConfig
from vllm.logger import init_logger as init_logger
from vllm.transformers_utils import processors as processors
from vllm.transformers_utils.gguf_utils import is_gguf as is_gguf
from vllm.transformers_utils.repo_utils import (
    get_hf_file_to_dict as get_hf_file_to_dict,
)
from vllm.transformers_utils.utils import (
    convert_model_repo_to_path as convert_model_repo_to_path,
)
from vllm.utils.func_utils import (
    get_allowed_kwarg_only_overrides as get_allowed_kwarg_only_overrides,
)

logger: Incomplete

class HashableDict(dict):
    def __hash__(self) -> int: ...

class HashableList(list):
    def __hash__(self) -> int: ...

def get_processor_cls_name_from_config(
    processor_name: str, revision: str | None = "main"
) -> str | None: ...
def get_processor(
    processor_name: str,
    *args: Any,
    revision: str | None = None,
    trust_remote_code: bool = False,
    processor_cls: type[_P] | tuple[type[_P], ...] = ...,
    **kwargs: Any,
) -> _P: ...

cached_get_processor: Incomplete

@lru_cache
def get_processor_kwargs_type(
    processor: ProcessorMixin,
) -> type[processing_utils.ProcessingKwargs]: ...
@lru_cache
def get_processor_kwargs_keys(
    kwargs_cls: type[processing_utils.ProcessingKwargs],
) -> set[str]: ...
def cached_get_processor_without_dynamic_kwargs(
    processor_name: str,
    *args: Any,
    revision: str | None = None,
    trust_remote_code: bool = False,
    processor_cls: type[_P] | tuple[type[_P], ...] = ...,
    **kwargs: Any,
) -> _P: ...
def cached_processor_from_config(
    model_config: ModelConfig,
    processor_cls: type[_P] | tuple[type[_P], ...] = ...,
    **kwargs: Any,
) -> _P: ...
def get_feature_extractor(
    processor_name: str,
    *args: Any,
    revision: str | None = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
): ...

cached_get_feature_extractor: Incomplete

def cached_feature_extractor_from_config(model_config: ModelConfig, **kwargs: Any): ...
def get_image_processor(
    processor_name: str,
    *args: Any,
    revision: str | None = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
): ...

cached_get_image_processor: Incomplete

def cached_image_processor_from_config(model_config: ModelConfig, **kwargs: Any): ...
def get_video_processor(
    processor_name: str,
    *args: Any,
    revision: str | None = None,
    trust_remote_code: bool = False,
    processor_cls_overrides: type[_V] | None = None,
    **kwargs: Any,
): ...

cached_get_video_processor: Incomplete

def cached_video_processor_from_config(
    model_config: ModelConfig, processor_cls: type[_V] | None = None, **kwargs: Any
): ...
