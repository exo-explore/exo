import abc
import torch
import torch.nn as nn
from .deepencoder import (
    DeepCLIPVisionTransformer as DeepCLIPVisionTransformer,
    build_sam_vit_b as build_sam_vit_b,
)
from .deepseek_vl2 import MlpProjector as MlpProjector
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import BatchFeature as BatchFeature
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
    NestedTensors as NestedTensors,
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
)
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config as cached_tokenizer_from_config
from vllm.transformers_utils.configs.deepseek_vl2 import (
    DeepseekVLV2Config as DeepseekVLV2Config,
)
from vllm.transformers_utils.processors.deepseek_ocr import (
    BASE_SIZE as BASE_SIZE,
    CROP_MODE as CROP_MODE,
    DeepseekOCRProcessor as DeepseekOCRProcessor,
    count_tiles as count_tiles,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor as AdapterLogitsProcessor,
    RequestLogitsProcessor as RequestLogitsProcessor,
)

IMAGE_SIZE: int

class DeepseekOCRImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, None]
    images_crop: Annotated[torch.Tensor, None]
    images_spatial_crop: Annotated[torch.Tensor, None]

class NoRepeatNGramLogitsProcessor:
    ngram_size: Incomplete
    window_size: Incomplete
    whitelist_token_ids: Incomplete
    def __init__(
        self,
        ngram_size: int,
        window_size: int,
        whitelist_token_ids: set[int] | None = None,
    ) -> None: ...
    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor: ...

class NGramPerReqLogitsProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, params: SamplingParams): ...
    def is_argmax_invariant(self) -> bool: ...
    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> RequestLogitsProcessor | None: ...

class DeepseekOCRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, cropping: bool = True
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...

class DeepseekOCRDummyInputsBuilder(BaseDummyInputsBuilder[DeepseekOCRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class DeepseekOCRMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekOCRProcessingInfo]
): ...

class DeepseekOCRForCausalLM(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    vision_config: Incomplete
    projector_config: Incomplete
    text_config: Incomplete
    image_token_id: Incomplete
    sam_model: Incomplete
    vision_model: Incomplete
    projector: Incomplete
    tile_tag: Incomplete
    global_view_pos: Incomplete
    image_newline: Incomplete
    view_seperator: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None: ...
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
    def get_mm_mapping(self) -> MultiModelKeys: ...
