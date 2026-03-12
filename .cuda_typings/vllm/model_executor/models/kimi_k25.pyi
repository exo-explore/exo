import abc
import torch
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from torch import nn
from transformers import BatchFeature
from transformers.processing_utils import ProcessorMixin
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig as CompressedTensorsConfig,
)
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector as KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel as MoonViT3dPretrainedModel,
    vision_tower_forward as vision_tower_forward,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
    NestedTensors as NestedTensors,
    VisionChunk as VisionChunk,
    VisionChunkImage as VisionChunkImage,
    VisionChunkVideo as VisionChunkVideo,
)
from vllm.multimodal.parse import (
    MultiModalDataItems as MultiModalDataItems,
    VisionChunkProcessorItems as VisionChunkProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    InputProcessingContext as InputProcessingContext,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs import KimiK25Config as KimiK25Config
from vllm.transformers_utils.processor import (
    cached_get_image_processor as cached_get_image_processor,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete

@dataclass
class MaxImageTokenMeta:
    width: int = ...
    height: int = ...

class KimiK25MediaPixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor | list[torch.Tensor], None]
    grid_thws: Annotated[torch.Tensor, None]

class MoonshotKimiVAutoProcessor(ProcessorMixin):
    attributes: Incomplete
    tokenizer_class: str
    media_processor: Incomplete
    media_token_id: Incomplete
    def __init__(
        self, media_processor=None, tokenizer=None, media_token_id: int | None = None
    ) -> None: ...
    def __call__(
        self,
        vision_chunks: list[VisionChunk] | None = None,
        *,
        text: list[int] | str,
        **kwargs,
    ) -> BatchFeature: ...

class KimiK25ProcessingInfo(BaseProcessingInfo):
    hf_config: Incomplete
    media_token_id: Incomplete
    media_processor: Incomplete
    hf_processor: Incomplete
    media_tokens_calculator: Incomplete
    def __init__(self, ctx: InputProcessingContext) -> None: ...
    def get_hf_processor(self): ...
    def get_hf_config(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...

class KimiK25DummyInputsBuilder(BaseDummyInputsBuilder[KimiK25ProcessingInfo]):
    media_token_id: Incomplete
    frame_per_chunk: Incomplete
    def __init__(self, info: KimiK25ProcessingInfo) -> None: ...
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_items(self): ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class KimiK25MultiModalProcessor(BaseMultiModalProcessor[KimiK25ProcessingInfo]):
    def split_video_chunks(self, video): ...

class KimiK25ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant, metaclass=abc.ABCMeta
):
    supports_encoder_tp_data: bool
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    use_data_parallel: Incomplete
    hidden_size: Incomplete
    device: Incomplete
    vision_tower: Incomplete
    mm_projector: Incomplete
    quant_config: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    media_placeholder: int
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
