import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
    split_list_into_ranges as split_list_into_ranges,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Iterator, Mapping
from transformers import PretrainedConfig as PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature as BatchFeature
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig as Qwen2_5OmniThinkerConfig,
)
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor,
)
from typing import Annotated, Any, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.forward_context import set_forward_context as set_forward_context
from vllm.logger import init_logger as init_logger
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImageEmbeddingInputs as Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs as Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs as Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLProcessingInfo as Qwen2_5_VLProcessingInfo,
    Qwen2_5_VLVideoEmbeddingInputs as Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs as Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs as Qwen2_5_VLVideoPixelInputs,
    Qwen2_5_VisionTransformer as Qwen2_5_VisionTransformer,
)
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioProcessingInfo as Qwen2AudioProcessingInfo,
)
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLMultiModalDataParser as Qwen2VLMultiModalDataParser,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem as ImageItem,
    ModalityData as ModalityData,
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems as AudioProcessorItems,
    DictEmbeddingItems as DictEmbeddingItems,
    ModalityDataItems as ModalityDataItems,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    ProcessorInputs as ProcessorInputs,
    TimingContext as TimingContext,
)
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    MultiModalPromptUpdates as MultiModalPromptUpdates,
    PlaceholderFeaturesInfo as PlaceholderFeaturesInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete

def check_interleaved_audio_video(
    is_video: torch.Tensor, is_audio: torch.Tensor, num_video: int, num_audio: int
) -> bool: ...
def merge_interleaved_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: MultiModalEmbeddings,
    is_video: torch.Tensor,
    is_audio: torch.Tensor,
    is_multimodal: torch.Tensor,
    num_video: int,
    num_audio: int,
) -> torch.Tensor: ...

class Qwen2_5OmniAudioFeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    input_features: Annotated[torch.Tensor | list[torch.Tensor], None]
    audio_feature_lengths: Annotated[torch.Tensor, None]
    feature_attention_mask: Annotated[torch.Tensor | list[torch.Tensor], None]

def create_qwen2_5_omni_thinker_field_factory(
    spatial_merge_size: int,
) -> Callable[[Mapping[str, torch.Tensor]], Mapping[str, MultiModalFieldConfig]]: ...

class Qwen2_5OmniThinkerMultiModalDataParser(Qwen2VLMultiModalDataParser):
    def __init__(self, spatial_merge_size: int, *args, **kwargs) -> None: ...

class Qwen2_5OmniThinkerProcessingInfo(
    Qwen2AudioProcessingInfo, Qwen2_5_VLProcessingInfo
):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> Qwen2_5OmniProcessor: ...
    def get_feature_extractor(self, **kwargs: object): ...
    def get_data_parser(self): ...
    def get_target_channels(self) -> int: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int] | None = None
    ) -> Mapping[str, int] | None: ...

class Qwen2_5OmniThinkerDummyInputsBuilder(
    BaseDummyInputsBuilder[Qwen2_5OmniThinkerProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Qwen2_5OmniThinkerMultiModalProcessor(
    BaseMultiModalProcessor[Qwen2_5OmniThinkerProcessingInfo]
):
    @classmethod
    def omni_get_updates_use_audio_in_video(
        cls,
        thinker_config: PretrainedConfig,
        audio_len: int,
        video_grid_thw: list[int] | torch.Tensor,
        video_second_per_grid_t: float,
    ) -> list[int]: ...

class Qwen2_5OmniConditionalGenerationMixin: ...

class Qwen2_5OmniThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
    SupportsMRoPE,
    Qwen2_5OmniConditionalGenerationMixin,
    metaclass=abc.ABCMeta,
):
    hf_to_vllm_mapper: Incomplete
    packed_modules_mapping: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    vllm_config: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    quant_config: Incomplete
    audio_tower: Incomplete
    visual: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def iter_mm_features(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, str, dict[str, Any]]]: ...
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
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
    def get_mm_mapping(self) -> MultiModelKeys: ...
