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
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import BatchFeature as BatchFeature
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.transformers.utils import (
    replace_linear_class as replace_linear_class,
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
    MultiModalProcessingInfo as MultiModalProcessingInfo,
    ProcessorInputs as ProcessorInputs,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    TimingContext as TimingContext,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config as cached_tokenizer_from_config
from vllm.transformers_utils.configs.deepseek_vl2 import (
    DeepseekVLV2Config as DeepseekVLV2Config,
    MlpProjectorConfig as MlpProjectorConfig,
    VisionEncoderConfig as VisionEncoderConfig,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)
from vllm.utils.torch_utils import set_default_torch_dtype as set_default_torch_dtype

class DeepseekVL2ImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, None]
    images_spatial_crop: Annotated[torch.Tensor, None]

class DeepseekVL2VImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], None]

DeepseekVL2ImageInputs: TypeAlias = (
    DeepseekVL2ImagePixelInputs | DeepseekVL2VImageEmbeddingInputs
)

class MlpProjector(nn.Module):
    cfg: Incomplete
    projector_type: Incomplete
    layers: Incomplete
    def __init__(self, cfg: MlpProjectorConfig) -> None: ...
    def forward(self, x): ...

class DeepseekVL2ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, cropping: bool = True
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...

class DeepseekVL2DummyInputsBuilder(BaseDummyInputsBuilder[DeepseekVL2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class DeepseekVL2MultiModalProcessor(
    BaseMultiModalProcessor[DeepseekVL2ProcessingInfo]
): ...

class DeepseekVLV2ForCausalLM(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    vision_config: Incomplete
    projector_config: Incomplete
    text_config: Incomplete
    image_token_id: int
    vision: Incomplete
    projector: Incomplete
    tile_tag: Incomplete
    global_view_pos: Incomplete
    image_newline: Incomplete
    view_seperator: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def patch_vit_for_tp(
        self, vit: torch.nn.Module, quant_config: QuantizationConfig
    ): ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
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
