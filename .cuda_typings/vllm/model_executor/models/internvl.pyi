import abc
import numpy.typing as npt
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
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from PIL import Image
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from transformers import (
    BatchFeature,
    PretrainedConfig as PretrainedConfig,
    TensorType as TensorType,
)
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.quantization.awq import AWQConfig as AWQConfig
from vllm.model_executor.models.intern_vit import (
    InternVisionModel as InternVisionModel,
    InternVisionPatchModel as InternVisionPatchModel,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.image import convert_image_mode as convert_image_mode
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
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

IMG_START: str
IMG_END: str
IMG_CONTEXT: str
IMAGENET_MEAN: Incomplete
IMAGENET_STD: Incomplete

class InternVLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values_flat: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

class InternVLImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], None]

InternVLImageInputs: TypeAlias = InternVLImagePixelInputs | InternVLImageEmbeddingInputs

class InternVLVideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]
    pixel_values_flat: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

class InternVLVideoEmbeddingInputs(TensorSchema):
    type: Literal["video_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], None]

InternVLVideoInputs: TypeAlias = InternVLVideoPixelInputs | InternVLVideoEmbeddingInputs

def build_transform(input_size: int): ...
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]: ...
def resolve_internvl_min_max_num(
    *,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]: ...
def get_internvl_target_ratios(min_num: int, max_num: int) -> list[tuple[int, int]]: ...
def calculate_internvl_targets(
    *,
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[int, int, int]: ...
def dynamic_preprocess_internvl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]: ...
def image_to_pixel_values_internvl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor: ...
def video_to_pixel_values_internvl(
    video: npt.NDArray,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor: ...

class BaseInternVLProcessor(ABC, metaclass=abc.ABCMeta):
    config: Incomplete
    tokenizer: Incomplete
    num_image_token: Incomplete
    image_size: Incomplete
    min_dynamic_patch: Incomplete
    max_dynamic_patch: Incomplete
    dynamic_image_size: Incomplete
    use_thumbnail: bool
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> None: ...
    @property
    @abstractmethod
    def image_token_id(self) -> int: ...
    @abstractmethod
    def get_image_repl(
        self, feature_size: int, num_patches: int | None
    ) -> PromptUpdateDetails[str]: ...
    def resolve_min_max_num(
        self,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        use_thumbnail: bool | None = None,
    ) -> tuple[int, int]: ...
    def resolve_target_ratios(
        self,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        use_thumbnail: bool | None = None,
    ) -> list[tuple[int, int]]: ...
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...
    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature: ...

class InternVLProcessor(BaseInternVLProcessor):
    video_token: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        video_token: str | None = None,
    ) -> None: ...
    @property
    def image_token_id(self) -> int: ...
    @property
    def video_token_id(self) -> int | None: ...
    @property
    def supports_video(self) -> bool: ...
    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        videos: npt.NDArray | list[npt.NDArray] | None = None,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature: ...
    def get_image_repl(
        self, feature_size: int, num_patches: int | None
    ) -> PromptUpdateDetails[str]: ...
    def get_video_repl(
        self,
        feature_size: int,
        num_patches: int | None = None,
        video_context_token: str = ...,
    ) -> PromptUpdateDetails[str]: ...

class BaseInternVLProcessingInfo(BaseProcessingInfo, metaclass=abc.ABCMeta):
    @abstractmethod
    def get_hf_processor(self, **kwargs: object) -> BaseInternVLProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, processor: BaseInternVLProcessor
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_max_image_tokens(self) -> int: ...

class BaseInternVLDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class BaseInternVLMultiModalProcessor(BaseMultiModalProcessor[_I]): ...

class InternVLProcessingInfo(BaseInternVLProcessingInfo):
    @property
    def supports_video(self): ...
    def get_supported_mm_limits(self): ...
    def get_video_token(self) -> str | None: ...
    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...
    def get_hf_processor(self, **kwargs: object) -> InternVLProcessor: ...

class InternVLDummyInputsBuilder(
    BaseInternVLDummyInputsBuilder[InternVLProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class InternVLMultiModalProcessor(
    BaseInternVLMultiModalProcessor[InternVLProcessingInfo]
): ...

class InternVLChatModel(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta
):
    supports_encoder_tp_data: bool
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    use_data_parallel: Incomplete
    patch_size: Incomplete
    patch_tokens: Incomplete
    num_image_token: Incomplete
    downsample_ratio: Incomplete
    ps_version: Incomplete
    is_mono: Incomplete
    vision_model: Incomplete
    mlp1: Incomplete
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
    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int: ...
    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int: ...
