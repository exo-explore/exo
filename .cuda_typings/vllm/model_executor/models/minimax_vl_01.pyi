import abc
import torch
import torch.nn as nn
from .clip import CLIPVisionModel as CLIPVisionModel
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .llava import (
    BaseLlavaMultiModalProcessor as BaseLlavaMultiModalProcessor,
    LlavaDummyInputsBuilder as LlavaDummyInputsBuilder,
    init_vision_tower_for_llava as init_vision_tower_for_llava,
)
from .llava_next import LlavaNextProcessingInfo as LlavaNextProcessingInfo
from .pixtral import PixtralHFVisionModel as PixtralHFVisionModel
from .siglip import SiglipVisionModel as SiglipVisionModel
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import BatchFeature as BatchFeature
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig as MultiModalFieldConfig
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class MiniMaxVL01ImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor | list[torch.Tensor], None]
    image_sizes: Annotated[torch.Tensor | None, None]

class MiniMaxVL01ImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, None]

MiniMaxVL01ImageInputs: TypeAlias = (
    MiniMaxVL01ImagePixelInputs | MiniMaxVL01ImageEmbeddingInputs
)

class MiniMaxVL01MultiModalProjector(nn.Module):
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

class MiniMaxVL01DummyInputsBuilder(LlavaDummyInputsBuilder): ...

class MiniMaxVL01ProcessingInfo(LlavaNextProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...

class MiniMaxVL01MultiModalProcessor(
    BaseLlavaMultiModalProcessor[MiniMaxVL01ProcessingInfo]
): ...

class MiniMaxVL01ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    vision_tower: Incomplete
    multi_modal_projector: Incomplete
    image_newline: Incomplete
    language_model: Incomplete
    vision_feature_layer: Incomplete
    vocab_size: Incomplete
    pad_token_id: int
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def pack_image_features(
        self, image_features: list[torch.Tensor], image_sizes: torch.Tensor
    ): ...
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
