import abc
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .qwen2_vl import Qwen2VisionTransformer as Qwen2VisionTransformer
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLVisionConfig as Qwen2VLVisionConfig,
)
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.logger import init_logger as init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
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
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete

class KananaVImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    vision_grid_thw: Annotated[torch.Tensor, None]

KananaVImageInputs: TypeAlias = KananaVImagePixelInputs

def build_pos_embeds(
    config: Qwen2VLVisionConfig, num_input_tokens: int, vision_hidden_size: int
) -> nn.Parameter | None: ...
def build_mlp(
    depth: int, hidden_size: int, output_hidden_size: int
) -> nn.Sequential: ...

class PatchMerge(nn.Module):
    merge_size: Incomplete
    def __init__(self, merge_size: int) -> None: ...
    def forward(self, x: torch.Tensor, channel_last: bool = False) -> torch.Tensor: ...

class DynamicCAbstractor(nn.Module):
    config: Incomplete
    merge_size: Incomplete
    pos_emb_size: Incomplete
    num_input_tokens: Incomplete
    pos_emb: Incomplete
    def __init__(self, config: Qwen2VLVisionConfig, num_input_tokens: int) -> None: ...
    net: Incomplete
    readout: Incomplete
    def build_net(self) -> None: ...
    def forward(
        self,
        flattened_visual_embeds: torch.Tensor,
        grid_thw: torch.Tensor,
        **unused_kwargs: object,
    ) -> BaseModelOutput: ...

class CustomQwen2VLVE(Qwen2VisionTransformer):
    def __init__(self, config: Qwen2VLVisionConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutput: ...
    def get_num_tokens(self) -> int: ...

class KananaVProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]: ...

class KananaVDummyInputsBuilder(BaseDummyInputsBuilder[KananaVProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class KananaVMultiModalProcessor(BaseMultiModalProcessor[KananaVProcessingInfo]):
    @property
    def media_token_id(self) -> int: ...

class KananaVForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    vision_model: Incomplete
    abstractor: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward_vision(
        self, pixel_values: torch.Tensor, image_metas: dict | None = None
    ) -> torch.Tensor: ...
    def forward_projector(
        self, visual_features: torch.Tensor, image_metas: dict | None = None
    ) -> torch.Tensor: ...
    def forward_and_project_vision(
        self, pixel_values: torch.Tensor, image_metas: dict | None = None
    ) -> torch.Tensor: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ): ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
