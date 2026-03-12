import abc
import torch
import torch.nn as nn
from .idefics2_vision_model import (
    Idefics2VisionTransformer as Idefics2VisionTransformer,
)
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
)
from .phi4mm_audio import AudioEmbedding as AudioEmbedding
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import (
    PretrainedConfig as PretrainedConfig,
    ProcessorMixin as ProcessorMixin,
    SequenceFeatureExtractor as SequenceFeatureExtractor,
)
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import get_pp_group as get_pp_group
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.models.llama import LlamaModel as LlamaModel
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
    NestedTensors as NestedTensors,
)
from vllm.multimodal.parse import (
    AudioProcessorItems as AudioProcessorItems,
    ImageEmbeddingItems as ImageEmbeddingItems,
    ImageProcessorItems as ImageProcessorItems,
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder as BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    ResolvedPromptUpdate as ResolvedPromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

SIGLIP_NAME: str
VISION_ENCODER_TO_PROCESSING_CONFIG: Incomplete

def get_navit_vision_model(layer_idx: int = -1, **kwargs): ...

class Phi4MMImageEncoder(nn.Module):
    layer_idx: Incomplete
    type_feature: Incomplete
    img_processor: Incomplete
    img_processor_padding: Incomplete
    num_img_tokens: Incomplete
    base_feat_height_target: Incomplete
    image_dim_out: Incomplete
    img_sizes: Incomplete
    image_attention_mask: Incomplete
    use_hd_transform: bool
    with_learnable_separator: bool
    hd_transform_order: str
    freeze_img_processor: bool
    crop_size: int
    image_token_compression_cls: str
    image_token_compression: Incomplete
    base_feat_height_reduction: int
    glb_GN: Incomplete
    sub_GN: Incomplete
    img_projection: Incomplete
    vocab_size: Incomplete
    img_features: Incomplete
    use_out_place_operations: bool
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
        model_dir: str = "",
    ) -> None: ...
    def get_img_features(
        self, img_embeds: torch.FloatTensor, attention_mask=None
    ) -> torch.FloatTensor: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        image_attention_mask: torch.Tensor,
    ) -> list[torch.FloatTensor]: ...

class Phi4MMImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor | list[torch.Tensor], None]
    image_sizes: Annotated[torch.Tensor, None]
    num_img_tokens: Annotated[list[int], None]
    image_attention_mask: Annotated[torch.Tensor, None]

class Phi4MMAudioFeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    audio_features: Annotated[torch.Tensor | list[torch.Tensor], None]

class Phi4MMAudioEmbeddingInputs(TensorSchema):
    type: Literal["audio_embeds"]
    data: Annotated[NestedTensors, None]

Phi4MMAudioInputs: TypeAlias = Phi4MMAudioFeatureInputs | Phi4MMAudioEmbeddingInputs

def cat_with_pad(tensors, dim, padding_value: int = 0): ...

class Phi4MMProcessingInfo(BaseProcessingInfo):
    @property
    def image_tokens(self) -> list[str]: ...
    @property
    def audio_tokens(self) -> list[str]: ...
    def get_dynamic_hd(self, processor: ProcessorMixin) -> int: ...
    def get_feature_extractor(self, **kwargs: object) -> SequenceFeatureExtractor: ...
    def get_data_parser(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, processor: ProcessorMixin
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_audio_num_frames(self, audio_len: int, sr: float) -> int: ...

class Phi4MMDummyInputsBuilder(BaseDummyInputsBuilder[Phi4MMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Phi4MMMultiModalProcessor(BaseMultiModalProcessor[Phi4MMProcessingInfo]): ...

class Phi4MMForCausalLM(
    nn.Module, SupportsLoRA, SupportsMultiModal, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    quant_config: Incomplete
    vision_encoder: Incomplete
    embed_tokens_extend: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
