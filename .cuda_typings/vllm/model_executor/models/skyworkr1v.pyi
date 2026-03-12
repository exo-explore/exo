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
from PIL import Image as Image
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import (
    BatchFeature,
    PretrainedConfig as PretrainedConfig,
    TensorType as TensorType,
)
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.quantization.awq import AWQConfig as AWQConfig
from vllm.model_executor.models.intern_vit import (
    InternVisionModel as InternVisionModel,
    InternVisionPatchModel as InternVisionPatchModel,
)
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

class SkyworkR1VImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values_flat: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

class SkyworkR1VImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], None]

SkyworkR1VImageInputs: TypeAlias = (
    SkyworkR1VImagePixelInputs | SkyworkR1VImageEmbeddingInputs
)

def build_transform(input_size: int): ...
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]: ...
def resolve_skyworkr1v_min_max_num(
    *,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]: ...
def get_skyworkr1v_target_ratios(
    min_num: int, max_num: int
) -> list[tuple[int, int]]: ...
def calculate_skyworkr1v_targets(
    *,
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[int, int, int]: ...
def dynamic_preprocess_skyworkr1v(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]: ...
def image_to_pixel_values_skyworkr1v(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor: ...

class SkyworkR1VProcessor:
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
    def image_token_id(self) -> int: ...
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

class SkyworkR1VProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> SkyworkR1VProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, processor: SkyworkR1VProcessor
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...

class SkyworkR1VDummyInputsBuilder(BaseDummyInputsBuilder[SkyworkR1VProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class SkyworkR1VMultiModalProcessor(
    BaseMultiModalProcessor[SkyworkR1VProcessingInfo]
): ...

class SkyworkR1VChatModel(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    patch_size: Incomplete
    num_image_token: Incomplete
    downsample_ratio: Incomplete
    ps_version: Incomplete
    is_mono: Incomplete
    vision_model: Incomplete
    mlp1: Incomplete
    language_model: Incomplete
    img_context_token_id: Incomplete
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
