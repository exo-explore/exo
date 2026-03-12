import abc
import torch
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import (
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model as run_dp_sharded_mrope_vision_model,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from torch import nn
from transformers import BatchFeature as BatchFeature
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from vllm.model_executor.models.moonvit import (
    MoonVitPretrainedModel as MoonVitPretrainedModel,
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
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs import (
    KimiVLConfig as KimiVLConfig,
    MoonViTConfig as MoonViTConfig,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

@dataclass
class MaxImageTokenMeta:
    width: int = ...
    height: int = ...

class KimiVLMultiModalProjector(nn.Module):
    use_data_parallel: Incomplete
    hidden_size: Incomplete
    pre_norm: Incomplete
    linear_1: Incomplete
    linear_2: Incomplete
    act: Incomplete
    def __init__(self, config: KimiVLConfig, prefix: str = "") -> None: ...
    def forward(self, image_features: torch.Tensor) -> torch.Tensor: ...

class KimiVLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor | list[torch.Tensor], None]
    image_grid_hws: Annotated[torch.Tensor, None]

KimiVLImageInputs = KimiVLImagePixelInputs

class KimiVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...
    @property
    def image_token_id(self) -> int: ...

class KimiVLDummyInputsBuilder(BaseDummyInputsBuilder[KimiVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class KimiVLMultiModalProcessor(BaseMultiModalProcessor[KimiVLProcessingInfo]): ...

class KimiVLForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    supports_encoder_tp_data: bool
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    quant_config: Incomplete
    use_data_parallel: Incomplete
    hidden_size: Incomplete
    vision_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    media_placeholder: int
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
