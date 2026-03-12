import abc
import torch
import torch.nn as nn
from .clip import CLIPVisionModel as CLIPVisionModel
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import (
    BatchFeature as BatchFeature,
    PretrainedConfig as PretrainedConfig,
    ProcessorMixin as ProcessorMixin,
)
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
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
from vllm.multimodal.processing import BaseDummyInputsBuilder as BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    MultiModalPromptUpdates as MultiModalPromptUpdates,
    PlaceholderFeaturesInfo as PlaceholderFeaturesInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    ResolvedPromptUpdate as ResolvedPromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete
CLIP_VIT_LARGE_PATCH14_336_CONFIG: Incomplete

class Phi3VImagePixelInputs(TensorSchema):
    type: Literal["pixel_values", "image_embeds"]
    pixel_values: Annotated[torch.Tensor | list[torch.Tensor], None]
    image_sizes: Annotated[torch.Tensor | None, None]

class Phi3VImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], None]

Phi3VImageInputs: TypeAlias = Phi3VImagePixelInputs | Phi3VImageEmbeddingInputs

class Phi3HDImageEmbedding(nn.Module):
    img_processor: Incomplete
    num_img_tokens: Incomplete
    image_dim_out: Incomplete
    use_hd_transform: Incomplete
    with_learnable_separator: Incomplete
    hd_transform_order: Incomplete
    glb_GN: Incomplete
    sub_GN: Incomplete
    img_projection: Incomplete
    type_feature: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None: ...
    def get_img_features(self, img_embeds: torch.FloatTensor) -> torch.FloatTensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, image_sizes: torch.Tensor
    ) -> torch.FloatTensor: ...
    def hd_feature_transform(self, image_features, image_sizes): ...
    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop): ...
    def add_image_newline(self, image_features_hd): ...

class Phi3VProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, processor: ProcessorMixin
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...

class Phi3VDummyInputsBuilder(BaseDummyInputsBuilder[Phi3VProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Phi3VMultiModalProcessor(BaseMultiModalProcessor[Phi3VProcessingInfo]): ...

class Phi3VForCausalLM(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    image_token_id: Incomplete
    embed_tokens: Incomplete
    vision_embed_tokens: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
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
    ): ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
