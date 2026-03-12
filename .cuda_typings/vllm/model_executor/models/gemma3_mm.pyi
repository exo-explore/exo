import abc
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
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
from transformers import BatchFeature as BatchFeature, Gemma3Config, Gemma3Processor
from transformers.models.gemma3.image_processing_gemma3 import (
    Gemma3ImageProcessor as Gemma3ImageProcessor,
)
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as GemmaRMSNorm
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
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
from vllm.multimodal.processing import BaseDummyInputsBuilder as BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    MultiModalPromptUpdates as MultiModalPromptUpdates,
    MultiModalPromptUpdatesApplyResult as MultiModalPromptUpdatesApplyResult,
    PlaceholderFeaturesInfo as PlaceholderFeaturesInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
    replace_token_matches as replace_token_matches,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete

class Gemma3ImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

Gemma3ImageInputs = Gemma3ImagePixelInputs

class Gemma3ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_crops(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Gemma3Processor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...
    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Gemma3Processor,
        mm_kwargs: Mapping[str, object],
    ) -> PromptUpdateDetails[str]: ...
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Gemma3Processor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...

class Gemma3DummyInputsBuilder(BaseDummyInputsBuilder[Gemma3ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Gemma3MultiModalProcessor(BaseMultiModalProcessor[Gemma3ProcessingInfo]): ...

class Gemma3MultiModalProjector(nn.Module):
    mm_input_projection_weight: Incomplete
    mm_soft_emb_norm: Incomplete
    patches_per_image: Incomplete
    tokens_per_side: Incomplete
    kernel_size: Incomplete
    avg_pool: Incomplete
    def __init__(self, config: Gemma3Config) -> None: ...
    def forward(self, vision_outputs: torch.Tensor): ...

class Gemma3ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    quant_config: Incomplete
    multimodal_config: Incomplete
    vision_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @property
    def dtype(self): ...
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
