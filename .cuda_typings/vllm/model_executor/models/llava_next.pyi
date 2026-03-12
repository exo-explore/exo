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
    BaseLlavaProcessingInfo as BaseLlavaProcessingInfo,
    LlavaDummyInputsBuilder as LlavaDummyInputsBuilder,
    LlavaLikeConfig as LlavaLikeConfig,
    LlavaMultiModalProjector as LlavaMultiModalProjector,
    init_vision_tower_for_llava as init_vision_tower_for_llava,
)
from .siglip import SiglipVisionModel as SiglipVisionModel
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import get_num_selected_vision_tokens as get_num_selected_vision_tokens
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import BatchFeature as BatchFeature
from typing import Annotated, Final, Literal, Protocol, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig as MultiModalFieldConfig
from vllm.multimodal.parse import ImageSize as ImageSize
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class LlavaNextImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor | list[torch.Tensor], None]
    image_sizes: Annotated[torch.Tensor | None, None]

class LlavaNextImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, None]

LlavaNextImageInputs: TypeAlias = (
    LlavaNextImagePixelInputs | LlavaNextImageEmbeddingInputs
)

class LlavaNextLikeConfig(LlavaLikeConfig, Protocol):
    image_grid_pinpoints: Final[list[list[int]]]

class LlavaNextProcessingInfo(BaseLlavaProcessingInfo):
    def get_hf_config(self) -> LlavaNextLikeConfig: ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...

class BaseLlavaNextMultiModalProcessor(
    BaseLlavaMultiModalProcessor[_I], metaclass=abc.ABCMeta
): ...
class LlavaNextMultiModalProcessor(
    BaseLlavaNextMultiModalProcessor[LlavaNextProcessingInfo]
): ...

class LlavaNextForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    select_layers: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    vision_tower: Incomplete
    image_newline: Incomplete
    multi_modal_projector: Incomplete
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
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
