import abc
import torch
import torch.nn as nn
from .clip import CLIPVisionModel as CLIPVisionModel
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsEagle3 as SupportsEagle3,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .module_mapping import MultiModelKeys as MultiModelKeys
from .pixtral import (
    PixtralHFEncoderInfo as PixtralHFEncoderInfo,
    PixtralHFVisionModel as PixtralHFVisionModel,
)
from .siglip import SiglipVisionModel as SiglipVisionModel
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    get_layer_index as get_layer_index,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import (
    get_num_selected_vision_tokens as get_num_selected_vision_tokens,
    get_vision_encoder_info as get_vision_encoder_info,
)
from _typeshed import Incomplete
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from transformers import (
    BatchFeature as BatchFeature,
    PretrainedConfig as PretrainedConfig,
)
from typing import Annotated, Final, Literal, Protocol, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.cache import (
    BaseMultiModalProcessorCache as BaseMultiModalProcessorCache,
)
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalInputs as MultiModalInputs,
    MultiModalKwargsItems as MultiModalKwargsItems,
    mm_inputs as mm_inputs,
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
    ProcessorInputs as ProcessorInputs,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
    TimingContext as TimingContext,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class LlavaImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]

class PixtralHFImagePixelInputs(TensorSchema):
    type: Literal["pixel_values_pixtral"]
    pixel_values: Annotated[torch.Tensor | list[torch.Tensor], None]

class LlavaImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, None]

LlavaImageInputs: TypeAlias = (
    LlavaImagePixelInputs | PixtralHFImagePixelInputs | LlavaImageEmbeddingInputs
)

class LlavaMultiModalProjector(nn.Module):
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

class LlavaLikeConfig(Protocol):
    vision_config: Final[PretrainedConfig]
    image_token_index: Final[int]
    vision_feature_select_strategy: Final[str]
    vision_feature_layer: Final[int | list[int]]

class LlavaLikeProcessor(Protocol):
    image_token: Final[str]

class BaseLlavaProcessingInfo(BaseProcessingInfo, metaclass=abc.ABCMeta):
    def get_hf_config(self) -> LlavaLikeConfig: ...
    def get_vision_encoder_info(self): ...
    @abstractmethod
    def get_hf_processor(self, **kwargs: object) -> LlavaLikeProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_max_image_tokens(self) -> int: ...

class LlavaDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class LlavaProcessingInfo(BaseLlavaProcessingInfo):
    def get_hf_processor(self, **kwargs: object): ...

class BaseLlavaMultiModalProcessor(
    BaseMultiModalProcessor[_I], metaclass=abc.ABCMeta
): ...
class LlavaMultiModalProcessor(BaseLlavaMultiModalProcessor[LlavaProcessingInfo]): ...

class PixtralHFProcessingInfo(BaseLlavaProcessingInfo):
    def get_hf_processor(self, **kwargs: object): ...

class PixtralHFMultiModalProcessor(
    BaseMultiModalProcessor[PixtralHFProcessingInfo]
): ...

def init_vision_tower_for_llava(
    hf_config: LlavaLikeConfig,
    quant_config: QuantizationConfig | None,
    *,
    require_post_norm: bool | None = None,
    prefix: str = "",
) -> CLIPVisionModel | SiglipVisionModel | PixtralHFVisionModel: ...

class LlavaForConditionalGeneration(
    nn.Module,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsEagle3,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]: ...
    config: Incomplete
    multimodal_config: Incomplete
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
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int: ...
    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int: ...

class MantisProcessingInfo(LlavaProcessingInfo):
    def get_hf_processor(self, **kwargs: object): ...

class MantisMultiModalProcessor(LlavaMultiModalProcessor):
    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInputs: ...

class MantisForConditionalGeneration(
    LlavaForConditionalGeneration, metaclass=abc.ABCMeta
): ...
