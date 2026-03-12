import asyncio
import numpy as np
import torch
import torch.nn as nn
from .interfaces_base import VllmModel as VllmModel
from _typeshed import Incomplete
from collections.abc import (
    AsyncGenerator,
    Callable as Callable,
    Iterable,
    Mapping,
    MutableSequence,
    Sequence,
)
from torch import Tensor
from typing import ClassVar, Literal, Protocol, TypeAlias, overload
from typing_extensions import Self, TypeIs
from vllm.config import (
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.inputs import TokensPrompt as TokensPrompt
from vllm.inputs.data import PromptType as PromptType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc as MambaStateCopyFunc,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.multimodal.inputs import MultiModalFeatureSpec as MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tasks import ScoreType as ScoreType
from vllm.utils.collection_utils import common_prefix as common_prefix
from vllm.utils.func_utils import supports_kw as supports_kw

logger: Incomplete
MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]

class SupportsMultiModal(Protocol):
    supports_multimodal: ClassVar[Literal[True]]
    supports_multimodal_raw_input_only: ClassVar[bool]
    supports_encoder_tp_data: ClassVar[bool]
    requires_raw_input_tokens: ClassVar[bool]
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def configure_mm_token_handling(self, vocab_size: int, mm_token_ids: list[int]): ...
    def get_language_model(self) -> VllmModel: ...
    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int: ...
    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int: ...
    @overload
    def embed_input_ids(self, input_ids: Tensor) -> Tensor: ...
    @overload
    def embed_input_ids(
        self,
        input_ids: Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        *,
        is_multimodal: torch.Tensor,
    ) -> Tensor: ...
    def embed_input_ids(
        self,
        input_ids: Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: Tensor | None = None,
    ) -> Tensor: ...

class SupportsMultiModalPruning(Protocol):
    supports_multimodal_pruning: ClassVar[Literal[True]]
    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ) -> tuple[MultiModalEmbeddings, Tensor, int]: ...

@overload
def supports_multimodal(model: type[object]) -> TypeIs[type[SupportsMultiModal]]: ...
@overload
def supports_multimodal(model: object) -> TypeIs[SupportsMultiModal]: ...
def supports_multimodal_raw_input_only(model: type[object] | object) -> bool: ...
def requires_raw_input_tokens(model: type[object] | object) -> bool: ...
def supports_multimodal_encoder_tp_data(model: type[object] | object) -> bool: ...
@overload
def supports_multimodal_pruning(
    model: type[object],
) -> TypeIs[type[SupportsMultiModalPruning]]: ...
@overload
def supports_multimodal_pruning(model: object) -> TypeIs[SupportsMultiModalPruning]: ...

class SupportsScoreTemplate(Protocol):
    supports_score_template: ClassVar[Literal[True]]
    @classmethod
    def get_score_template(cls, query: str, document: str) -> str | None: ...
    @classmethod
    def post_process_tokens(cls, prompt: TokensPrompt) -> None: ...

@overload
def supports_score_template(
    model: type[object],
) -> TypeIs[type[SupportsScoreTemplate]]: ...
@overload
def supports_score_template(model: object) -> TypeIs[SupportsScoreTemplate]: ...

class SupportsLoRA(Protocol):
    supports_lora: ClassVar[Literal[True]]
    is_3d_moe_weight: ClassVar[bool]
    is_non_gated_moe: ClassVar[bool]
    embedding_modules: ClassVar[dict[str, str]]
    packed_modules_mapping: dict[str, list[str]]
    lora_skip_prefixes: ClassVar[list[str]]

class _SupportsLoRAType(Protocol):
    supports_lora: Literal[True]
    packed_modules_mapping: dict[str, list[str]]
    embedding_modules: dict[str, str]

@overload
def supports_lora(model: type[object]) -> TypeIs[type[SupportsLoRA]]: ...
@overload
def supports_lora(model: object) -> TypeIs[SupportsLoRA]: ...

class SupportsPP(Protocol):
    supports_pp: ClassVar[Literal[True]]
    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors: ...
    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        *,
        intermediate_tensors: IntermediateTensors | None,
    ) -> IntermediateTensors | None: ...

class _SupportsPPType(Protocol):
    supports_pp: Literal[True]
    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors: ...
    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        *,
        intermediate_tensors: IntermediateTensors | None,
    ) -> Tensor | IntermediateTensors: ...

@overload
def supports_pp(model: type[object]) -> TypeIs[type[SupportsPP]]: ...
@overload
def supports_pp(model: object) -> TypeIs[SupportsPP]: ...

class HasInnerState(Protocol):
    has_inner_state: ClassVar[Literal[True]]

@overload
def has_inner_state(model: object) -> TypeIs[HasInnerState]: ...
@overload
def has_inner_state(model: type[object]) -> TypeIs[type[HasInnerState]]: ...

class IsAttentionFree(Protocol):
    is_attention_free: ClassVar[Literal[True]]

@overload
def is_attention_free(model: object) -> TypeIs[IsAttentionFree]: ...
@overload
def is_attention_free(model: type[object]) -> TypeIs[type[IsAttentionFree]]: ...

class IsHybrid(Protocol):
    is_hybrid: ClassVar[Literal[True]]
    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]: ...
    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]: ...

@overload
def is_hybrid(model: object) -> TypeIs[IsHybrid]: ...
@overload
def is_hybrid(model: type[object]) -> TypeIs[type[IsHybrid]]: ...

class MixtureOfExperts(Protocol):
    expert_weights: MutableSequence[Sequence[Tensor]]
    num_moe_layers: int
    num_expert_groups: int
    num_logical_experts: int
    num_physical_experts: int
    num_local_physical_experts: int
    num_routed_experts: int
    num_shared_experts: int
    num_redundant_experts: int
    moe_layers: Iterable[nn.Module]
    def set_eplb_state(
        self,
        expert_load_view: Tensor,
        logical_to_physical_map: Tensor,
        logical_replica_count: Tensor,
    ) -> None: ...
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...

def is_mixture_of_experts(model: object) -> TypeIs[MixtureOfExperts]: ...

class HasNoOps(Protocol):
    has_noops: ClassVar[Literal[True]]

@overload
def has_noops(model: object) -> TypeIs[HasNoOps]: ...
@overload
def has_noops(model: type[object]) -> TypeIs[type[HasNoOps]]: ...

class SupportsMambaPrefixCaching(Protocol):
    supports_mamba_prefix_caching: ClassVar[Literal[True]]

@overload
def supports_mamba_prefix_caching(
    model: object,
) -> TypeIs[SupportsMambaPrefixCaching]: ...
@overload
def supports_mamba_prefix_caching(
    model: type[object],
) -> TypeIs[type[SupportsMambaPrefixCaching]]: ...

class SupportsCrossEncoding(Protocol):
    score_type: ClassVar[ScoreType]

class SupportsLateInteraction(Protocol):
    score_type: ClassVar[ScoreType]

class SupportsQuant:
    hf_to_vllm_mapper: ClassVar[WeightsMapper | None]
    packed_modules_mapping: ClassVar[dict[str, list[str]] | None]
    quant_config: QuantizationConfig | None
    def __new__(cls, *args, **kwargs) -> Self: ...

class SupportsRealtime(Protocol):
    supports_realtime: ClassVar[Literal[True]]
    realtime_max_tokens: ClassVar[int]
    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]: ...

@overload
def supports_realtime(model: type[object]) -> TypeIs[type[SupportsRealtime]]: ...
@overload
def supports_realtime(model: object) -> TypeIs[SupportsRealtime]: ...

class SupportsTranscription(Protocol):
    supported_languages: ClassVar[Mapping[str, str]]
    supports_transcription: ClassVar[Literal[True]]
    supports_transcription_only: ClassVar[bool]
    supports_segment_timestamp: ClassVar[bool]
    supports_explicit_language_detection: ClassVar[bool]
    def __init_subclass__(cls, **kwargs) -> None: ...
    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType: ...
    @classmethod
    def get_other_languages(cls) -> Mapping[str, str]: ...
    @classmethod
    def validate_language(cls, language: str | None) -> str | None: ...
    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: Literal["transcribe", "translate"]
    ) -> SpeechToTextConfig: ...
    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None: ...
    @classmethod
    def post_process_output(cls, text: str) -> str: ...
    @classmethod
    def get_language_detection_prompt(
        cls, audio: np.ndarray, stt_config: SpeechToTextConfig
    ) -> PromptType: ...
    @classmethod
    def parse_language_detection_output(
        cls, token_ids: list[int], tokenizer: object
    ) -> str: ...
    @classmethod
    def get_language_token_ids(cls, tokenizer: object) -> list[int] | None: ...

@overload
def supports_transcription(
    model: type[object],
) -> TypeIs[type[SupportsTranscription]]: ...
@overload
def supports_transcription(model: object) -> TypeIs[SupportsTranscription]: ...

class SupportsEagleBase(Protocol):
    has_own_lm_head: bool
    has_own_embed_tokens: bool

@overload
def supports_any_eagle(model: type[object]) -> TypeIs[type[SupportsEagleBase]]: ...
@overload
def supports_any_eagle(model: object) -> TypeIs[SupportsEagleBase]: ...

class SupportsEagle(SupportsEagleBase, Protocol):
    supports_eagle: ClassVar[Literal[True]]

@overload
def supports_eagle(model: type[object]) -> TypeIs[type[SupportsEagle]]: ...
@overload
def supports_eagle(model: object) -> TypeIs[SupportsEagle]: ...

class SupportsEagle3(SupportsEagleBase, Protocol):
    supports_eagle3: ClassVar[Literal[True]]
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]: ...

@overload
def supports_eagle3(model: type[object]) -> TypeIs[type[SupportsEagle3]]: ...
@overload
def supports_eagle3(model: object) -> TypeIs[SupportsEagle3]: ...

class SupportsMRoPE(Protocol):
    supports_mrope: ClassVar[Literal[True]]
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list["MultiModalFeatureSpec"]
    ) -> tuple[torch.Tensor, int]: ...

@overload
def supports_mrope(model: type[object]) -> TypeIs[type[SupportsMRoPE]]: ...
@overload
def supports_mrope(model: object) -> TypeIs[SupportsMRoPE]: ...

class SupportsXDRoPE(Protocol):
    supports_xdrope: ClassVar[Literal[True]]
    def get_xdrope_input_positions(
        self, input_tokens: list[int], mm_features: list["MultiModalFeatureSpec"]
    ) -> torch.Tensor: ...

@overload
def supports_xdrope(model: type[object]) -> TypeIs[type[SupportsXDRoPE]]: ...
@overload
def supports_xdrope(model: object) -> TypeIs[SupportsXDRoPE]: ...
