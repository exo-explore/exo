import abc
import torch.nn as nn
from .interfaces import (
    has_inner_state as has_inner_state,
    has_noops as has_noops,
    is_attention_free as is_attention_free,
    is_hybrid as is_hybrid,
    requires_raw_input_tokens as requires_raw_input_tokens,
    supports_mamba_prefix_caching as supports_mamba_prefix_caching,
    supports_multimodal as supports_multimodal,
    supports_multimodal_encoder_tp_data as supports_multimodal_encoder_tp_data,
    supports_multimodal_raw_input_only as supports_multimodal_raw_input_only,
    supports_pp as supports_pp,
    supports_transcription as supports_transcription,
)
from .interfaces_base import (
    get_attn_type as get_attn_type,
    get_default_seq_pooling_type as get_default_seq_pooling_type,
    get_default_tok_pooling_type as get_default_tok_pooling_type,
    get_score_type as get_score_type,
    is_pooling_model as is_pooling_model,
    is_text_generation_model as is_text_generation_model,
)
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable as Callable, Set as Set
from dataclasses import dataclass, field
from vllm import envs as envs
from vllm.config import (
    ModelConfig as ModelConfig,
    iter_architecture_defaults as iter_architecture_defaults,
    try_match_architecture_defaults as try_match_architecture_defaults,
)
from vllm.config.model import AttnTypeStr as AttnTypeStr
from vllm.config.pooler import (
    SequencePoolingType as SequencePoolingType,
    TokenPoolingType as TokenPoolingType,
)
from vllm.logger import init_logger as init_logger
from vllm.logging_utils import logtime as logtime
from vllm.tasks import ScoreType as ScoreType
from vllm.transformers_utils.dynamic_module import (
    try_get_class_from_dynamic_module as try_get_class_from_dynamic_module,
)
from vllm.utils.hashing import safe_hash as safe_hash

logger: Incomplete

@dataclass(frozen=True)
class _ModelInfo:
    architecture: str
    is_text_generation_model: bool
    is_pooling_model: bool
    attn_type: AttnTypeStr
    default_seq_pooling_type: SequencePoolingType
    default_tok_pooling_type: TokenPoolingType
    score_type: ScoreType
    supports_multimodal: bool
    supports_multimodal_raw_input_only: bool
    requires_raw_input_tokens: bool
    supports_multimodal_encoder_tp_data: bool
    supports_pp: bool
    has_inner_state: bool
    is_attention_free: bool
    is_hybrid: bool
    has_noops: bool
    supports_mamba_prefix_caching: bool
    supports_transcription: bool
    supports_transcription_only: bool
    @staticmethod
    def from_model_cls(model: type[nn.Module]) -> _ModelInfo: ...

class _BaseRegisteredModel(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def inspect_model_cls(self) -> _ModelInfo: ...
    @abstractmethod
    def load_model_cls(self) -> type[nn.Module]: ...

@dataclass(frozen=True)
class _RegisteredModel(_BaseRegisteredModel):
    interfaces: _ModelInfo
    model_cls: type[nn.Module]
    @staticmethod
    def from_model_cls(model_cls: type[nn.Module]): ...
    def inspect_model_cls(self) -> _ModelInfo: ...
    def load_model_cls(self) -> type[nn.Module]: ...

@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    module_name: str
    class_name: str
    def inspect_model_cls(self) -> _ModelInfo: ...
    def load_model_cls(self) -> type[nn.Module]: ...

@dataclass
class _ModelRegistry:
    models: dict[str, _BaseRegisteredModel] = field(default_factory=dict)
    def get_supported_archs(self) -> Set[str]: ...
    def register_model(
        self, model_arch: str, model_cls: type[nn.Module] | str
    ) -> None: ...
    def inspect_model_cls(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> tuple[_ModelInfo, str]: ...
    def resolve_model_cls(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> tuple[type[nn.Module], str]: ...
    def is_text_generation_model(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> bool: ...
    def is_pooling_model(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> bool: ...
    def is_multimodal_model(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> bool: ...
    def is_multimodal_raw_input_only_model(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> bool: ...
    def is_pp_supported_model(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> bool: ...
    def model_has_inner_state(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> bool: ...
    def is_attention_free_model(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> bool: ...
    def is_hybrid_model(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> bool: ...
    def is_noops_model(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> bool: ...
    def is_transcription_model(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> bool: ...
    def is_transcription_only_model(
        self, architectures: str | list[str], model_config: ModelConfig
    ) -> bool: ...

ModelRegistry: Incomplete
