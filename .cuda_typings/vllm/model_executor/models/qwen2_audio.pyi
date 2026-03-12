import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers.models.qwen2_audio import Qwen2AudioProcessor
from transformers.models.whisper import WhisperFeatureExtractor
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem as AudioItem,
    ModalityData as ModalityData,
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems as AudioProcessorItems,
    DictEmbeddingItems as DictEmbeddingItems,
    ModalityDataItems as ModalityDataItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class Qwen2AudioFeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    input_features: Annotated[torch.Tensor | list[torch.Tensor], None]
    feature_attention_mask: Annotated[torch.Tensor, None]

class Qwen2AudioEmbeddingInputs(TensorSchema):
    type: Literal["audio_embeds"]
    audio_embeds: Annotated[list[torch.Tensor], None]

Qwen2AudioInputs: TypeAlias = Qwen2AudioFeatureInputs | Qwen2AudioEmbeddingInputs

class Qwen2AudioMultiModalProjector(nn.Module):
    linear: Incomplete
    def __init__(self, audio_hidden_size: int, text_hidden_size: int) -> None: ...
    def forward(self, audio_features): ...

class Qwen2AudioMultiModalDataParser(MultiModalDataParser): ...

class Qwen2AudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> Qwen2AudioProcessor: ...
    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor: ...
    def get_data_parser(self): ...
    def get_target_channels(self) -> int: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int] | None = None
    ) -> Mapping[str, int]: ...

class Qwen2AudioDummyInputsBuilder(BaseDummyInputsBuilder[Qwen2AudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Qwen2AudioMultiModalProcessor(
    BaseMultiModalProcessor[Qwen2AudioProcessingInfo]
): ...

class Qwen2AudioForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    quant_config: Incomplete
    audio_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
