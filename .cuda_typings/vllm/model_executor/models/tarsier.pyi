import abc
import torch
import torch.nn as nn
from .clip import CLIPVisionModel as CLIPVisionModel
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .siglip import SiglipVisionModel as SiglipVisionModel
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    get_layer_index as get_layer_index,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import (
    VisionEncoderInfo as VisionEncoderInfo,
    get_num_selected_vision_tokens as get_num_selected_vision_tokens,
    get_vision_encoder_info as get_vision_encoder_info,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import BatchFeature, PretrainedConfig as PretrainedConfig
from transformers.image_utils import ImageInput as ImageInput
from transformers.models.llava import LlavaProcessor
from transformers.processing_utils import ProcessingKwargs, Unpack as Unpack
from transformers.tokenization_utils_base import (
    PreTokenizedInput as PreTokenizedInput,
    TextInput as TextInput,
)
from typing import Annotated, Final, Literal, Protocol, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.llava import (
    LlavaDummyInputsBuilder as LlavaDummyInputsBuilder,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.cache import (
    BaseMultiModalProcessorCache as BaseMultiModalProcessorCache,
)
from vllm.multimodal.inputs import (
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
    InputProcessingContext as InputProcessingContext,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class TarsierImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]

class TarsierImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, None]

TarsierImageInputs: TypeAlias = TarsierImagePixelInputs | TarsierImageEmbeddingInputs

class TarsierHfConfig(Protocol):
    vision_config: Final[PretrainedConfig]
    text_config: Final[PretrainedConfig]
    image_token_index: Final[int]
    vision_feature_select_strategy: Final[str]
    vision_feature_layer: Final[int | list[int]]
    projector_hidden_act: Final[str]
    image_newline_idx: Final[int]
    image_new_idx: Final[int]
    multimodal_projector_bias: bool

class TarsierProcessorKwargs(ProcessingKwargs, total=False): ...

class TarsierProcessor(LlavaProcessor):
    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[TarsierProcessorKwargs],
    ) -> BatchFeature: ...

class TarsierMultiModalProjector(nn.Module):
    linear_1: Incomplete
    act: Incomplete
    linear_2: Incomplete
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_act: str,
        multimodal_projector_bias: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, image_features: torch.Tensor) -> torch.Tensor: ...

class TarsierProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> TarsierHfConfig: ...
    def get_vision_encoder_info(self) -> VisionEncoderInfo: ...
    def get_hf_processor(self, **kwargs: object) -> TarsierProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_max_image_tokens(self) -> int: ...
    def get_image_newline_idx(self) -> int: ...
    def get_image_new_idx(self) -> int: ...

class TarsierDummyInputsBuilder(LlavaDummyInputsBuilder[_I_Tarsier]): ...
class TarsierMultiModalProcessor(BaseMultiModalProcessor[_I_Tarsier]): ...

def init_vision_tower_for_tarsier(
    hf_config: TarsierHfConfig,
    quant_config: QuantizationConfig | None,
    *,
    require_post_norm: bool | None = None,
    prefix: str = "",
) -> CLIPVisionModel | SiglipVisionModel: ...

class TarsierForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    vision_tower: Incomplete
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
