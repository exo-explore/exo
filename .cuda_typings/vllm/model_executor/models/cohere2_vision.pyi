import abc
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
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
from torch import nn
from transformers import (
    BatchFeature as BatchFeature,
    PretrainedConfig as PretrainedConfig,
)
from transformers.models.cohere2_vision import Cohere2VisionConfig
from transformers.models.cohere2_vision.image_processing_cohere2_vision_fast import (
    Cohere2VisionImageProcessorFast as Cohere2VisionImageProcessorFast,
)
from transformers.models.cohere2_vision.processing_cohere2_vision import (
    Cohere2VisionProcessor,
)
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.activation import MulAndSilu as MulAndSilu
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.quantization.awq import AWQConfig as AWQConfig
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
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
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class Cohere2VisionImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

class Cohere2VisionMultiModalProjector(nn.Module):
    downsample_factor: Incomplete
    intermediate_size: Incomplete
    linear_1: Incomplete
    act: Incomplete
    linear_2: Incomplete
    def __init__(self, config: Cohere2VisionConfig, prefix: str = "") -> None: ...
    def forward(self, image_features): ...
    def pixel_shuffle(self, image_features: torch.Tensor) -> torch.Tensor: ...

class Cohere2VisionProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Cohere2VisionConfig: ...
    def get_hf_processor(self, **kwargs: object) -> Cohere2VisionProcessor: ...
    def get_image_processor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_num_patches(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Cohere2VisionProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...

class Cohere2VisionDummyInputsBuilder(
    BaseDummyInputsBuilder[Cohere2VisionProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Cohere2VisionMultiModalProcessor(
    BaseMultiModalProcessor[Cohere2VisionProcessingInfo]
): ...

class Cohere2VisionForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    config: Incomplete
    quant_config: Incomplete
    multimodal_config: Incomplete
    vision_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @property
    def dtype(self): ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
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
