import abc
from _typeshed import Incomplete
from abc import abstractmethod
from collections.abc import Callable as Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property as cached_property
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from typing import Any, overload
from vllm.config import ModelConfig as ModelConfig
from vllm.logger import init_logger as init_logger
from vllm.multimodal.inputs import MultiModalDataDict as MultiModalDataDict
from vllm.multimodal.parse import (
    DictEmbeddingItems as DictEmbeddingItems,
    EmbeddingItems as EmbeddingItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.transformers_utils.processor import (
    cached_processor_from_config as cached_processor_from_config,
)
from vllm.utils.func_utils import (
    get_allowed_kwarg_only_overrides as get_allowed_kwarg_only_overrides,
)
from vllm.utils.jsontree import JSONTree as JSONTree, json_map_leaves as json_map_leaves
from vllm.utils.mistral import is_mistral_tokenizer as is_mistral_tokenizer

logger: Incomplete

@dataclass
class TimingContext:
    enabled: bool = ...
    stage_secs: dict[str, float] = field(default_factory=dict)
    @property
    def total_secs(self) -> float: ...
    @contextmanager
    def record(self, stage: str): ...
    def get_stats_dict(self): ...

@dataclass(frozen=True)
class InputProcessingContext:
    model_config: ModelConfig
    tokenizer: TokenizerLike | None
    def get_tokenizer(self) -> TokenizerLike: ...
    @overload
    def get_hf_config(self, /) -> PretrainedConfig: ...
    @overload
    def get_hf_config(self, typ: type[_C] | tuple[type[_C], ...], /) -> _C: ...
    def get_hf_image_processor_config(self) -> dict[str, Any]: ...
    def get_mm_config(self): ...
    @overload
    def get_hf_processor(self, /, **kwargs: object) -> ProcessorMixin: ...
    @overload
    def get_hf_processor(
        self, typ: type[_P] | tuple[type[_P], ...], /, **kwargs: object
    ) -> _P: ...
    def init_processor(self, typ: type[_T], /, **kwargs: object) -> _T: ...
    def get_merged_mm_kwargs(self, kwargs: Mapping[str, object]): ...
    def call_hf_processor(
        self,
        hf_processor: Callable[..., BatchFeature] | ProcessorMixin,
        data: Mapping[str, object],
        kwargs: Mapping[str, object] = {},
        *,
        num_tries: int = 1,
        max_tries: int = 5,
    ) -> BatchFeature: ...

class BaseProcessingInfo(metaclass=abc.ABCMeta):
    ctx: Incomplete
    def __init__(self, ctx: InputProcessingContext) -> None: ...
    @property
    def model_id(self) -> str: ...
    def get_tokenizer(self) -> TokenizerLike: ...
    def get_hf_config(self) -> PretrainedConfig: ...
    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin: ...
    def get_default_tok_params(self) -> TokenizeParams: ...
    @cached_property
    def default_tok_params(self) -> TokenizeParams: ...
    def get_data_parser(self) -> MultiModalDataParser: ...
    @cached_property
    def data_parser(self) -> MultiModalDataParser: ...
    @property
    def skip_prompt_length_check(self) -> bool: ...
    @abstractmethod
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    @cached_property
    def supported_mm_limits(self) -> Mapping[str, int | None]: ...
    @cached_property
    def allowed_mm_limits(self) -> Mapping[str, int]: ...
    def validate_num_items(self, modality: str, num_items: int) -> None: ...
    def parse_mm_data(
        self, mm_data: MultiModalDataDict, *, validate: bool = True
    ) -> MultiModalDataItems: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None: ...
