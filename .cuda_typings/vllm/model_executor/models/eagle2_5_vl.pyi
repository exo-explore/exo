import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .internvl import (
    BaseInternVLDummyInputsBuilder as BaseInternVLDummyInputsBuilder,
    BaseInternVLMultiModalProcessor as BaseInternVLMultiModalProcessor,
    BaseInternVLProcessingInfo as BaseInternVLProcessingInfo,
    BaseInternVLProcessor as BaseInternVLProcessor,
    IMG_CONTEXT as IMG_CONTEXT,
    IMG_END as IMG_END,
    IMG_START as IMG_START,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import PretrainedConfig as PretrainedConfig
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.siglip import SiglipVisionModel as SiglipVisionModel
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.processing import PromptUpdateDetails as PromptUpdateDetails
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class Eagle2_5_VLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values_flat: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

class Eagle2_5_VLImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], None]

Eagle2_5_VLImageInputs: TypeAlias = (
    Eagle2_5_VLImagePixelInputs | Eagle2_5_VLImageEmbeddingInputs
)

class Eagle2_5_VLProcessor(BaseInternVLProcessor):
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

class Eagle2_5_VLProcessingInfo(BaseInternVLProcessingInfo):
    def get_hf_processor(self, **kwargs) -> Eagle2_5_VLProcessor: ...

class Eagle2_5_VLDummyInputsBuilder(
    BaseInternVLDummyInputsBuilder[Eagle2_5_VLProcessingInfo]
): ...
class Eagle2_5_VLMultiModalProcessor(
    BaseInternVLMultiModalProcessor[Eagle2_5_VLProcessingInfo]
): ...

class Eagle2_5_VLForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta
):
    supports_encoder_tp_data: bool
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    use_data_parallel: Incomplete
    patch_size: Incomplete
    downsample_ratio: Incomplete
    num_image_token: Incomplete
    select_layer: Incomplete
    vision_model: Incomplete
    mlp1: Incomplete
    language_model: Incomplete
    img_context_token_id: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def pixel_shuffle(
        self, x: torch.Tensor, scale_factor: float = 0.5
    ) -> torch.Tensor: ...
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
    def get_mm_mapping(self) -> MultiModelKeys: ...
