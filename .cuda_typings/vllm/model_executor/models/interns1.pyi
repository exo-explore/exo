import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import InternVLProcessor, PretrainedConfig as PretrainedConfig
from transformers.models.got_ocr2.image_processing_got_ocr2_fast import (
    GotOcr2ImageProcessorFast as GotOcr2ImageProcessorFast,
)
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.interns1_vit import (
    InternS1VisionModel as InternS1VisionModel,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems as ImageEmbeddingItems,
    ImageProcessorItems as ImageProcessorItems,
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.processor import (
    cached_video_processor_from_config as cached_video_processor_from_config,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class InternS1MultiModalProjector(nn.Module):
    layer_norm: Incomplete
    linear_1: Incomplete
    act: Incomplete
    linear_2: Incomplete
    def __init__(self, config) -> None: ...
    def forward(self, image_features): ...

class InternS1ImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

class InternS1ImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], None]

InternS1ImageInputs: TypeAlias = InternS1ImagePixelInputs | InternS1ImageEmbeddingInputs

class InternS1VideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]
    pixel_values: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

class InternS1VideoEmbeddingInputs(TensorSchema):
    type: Literal["video_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], None]

InternS1VideoInputs: TypeAlias = InternS1VideoPixelInputs | InternS1VideoEmbeddingInputs

def resolve_interns1_min_max_num(
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]: ...
def get_interns1_target_ratios(min_num: int, max_num: int) -> list[tuple[int, int]]: ...

class InternS1ProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> InternVLProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: InternVLProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...
    def resolve_target_ratios(self, use_thumbnail: bool | None = None): ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_max_image_tokens(self) -> int: ...
    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...

class InternS1DummyInputsBuilder(BaseDummyInputsBuilder[InternS1ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class InternS1MultiModalProcessor(BaseMultiModalProcessor[InternS1ProcessingInfo]): ...

class InternS1ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    patch_size: Incomplete
    num_image_token: Incomplete
    downsample_ratio: Incomplete
    vision_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    img_context_token_id: Incomplete
    video_context_token_id: Incomplete
    visual_token_mask: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def pixel_shuffle(self, x, scale_factor: float = 0.5): ...
    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
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
    ) -> IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
