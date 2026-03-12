import abc
import torch
import torch.nn as nn
from .clip import CLIPVisionModel as CLIPVisionModel
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .llava import (
    LlavaDummyInputsBuilder as LlavaDummyInputsBuilder,
    init_vision_tower_for_llava as init_vision_tower_for_llava,
)
from .llava_next import (
    BaseLlavaNextMultiModalProcessor as BaseLlavaNextMultiModalProcessor,
    LlavaNextLikeConfig as LlavaNextLikeConfig,
    LlavaNextProcessingInfo as LlavaNextProcessingInfo,
)
from .siglip import SiglipVisionModel as SiglipVisionModel
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import LlavaOnevisionConfig
from typing import Annotated, Final, Literal, Protocol, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
    VideoEmbeddingItems as VideoEmbeddingItems,
    VideoProcessorItems as VideoProcessorItems,
)
from vllm.multimodal.processing import (
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class LlavaOnevisionVideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: Annotated[torch.Tensor | list[torch.Tensor], None]

class LlavaOnevisionImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor | list[torch.Tensor], None]
    image_sizes: Annotated[torch.Tensor | None, None]

class LlavaOnevisionImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, None]

LlavaOnevisionImageInputs: TypeAlias = (
    LlavaOnevisionImagePixelInputs | LlavaOnevisionImageEmbeddingInputs
)
LlavaOnevisionMultiInputs: TypeAlias = (
    LlavaOnevisionImageInputs | LlavaOnevisionVideoPixelInputs
)

class LlavaOnevisionLikeConfig(LlavaNextLikeConfig, Protocol):
    video_token_index: Final[int]

class LlavaOnevisionProcessingInfo(LlavaNextProcessingInfo):
    def get_hf_config(self) -> LlavaOnevisionLikeConfig: ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_num_video_tokens(
        self, *, image_width: int, image_height: int, num_frames: int
    ) -> int: ...
    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...
    def get_max_video_tokens(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...

class LlavaOnevisionDummyInputsBuilder(
    LlavaDummyInputsBuilder[LlavaOnevisionProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class LlavaOnevisionMultiModalProcessor(
    BaseLlavaNextMultiModalProcessor[LlavaOnevisionProcessingInfo]
): ...

class LlavaOnevisionMultiModalProjector(nn.Module):
    linear_1: Incomplete
    act: Incomplete
    linear_2: Incomplete
    def __init__(self, config: LlavaOnevisionConfig) -> None: ...
    def forward(self, image_features: torch.Tensor) -> torch.Tensor: ...

class LlavaOnevisionForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    vision_tower: Incomplete
    image_newline: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def apply_pooling(self, image_features: torch.Tensor, stride: int = 2): ...
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
