import abc
import torch
import torch.nn as nn
from .clip import CLIPVisionModel as CLIPVisionModel
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .siglip import SiglipVisionModel as SiglipVisionModel
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    flatten_bn as flatten_bn,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import get_vision_encoder_info as get_vision_encoder_info
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import BatchFeature as BatchFeature
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.cache import (
    BaseMultiModalProcessorCache as BaseMultiModalProcessorCache,
)
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
    InputProcessingContext as InputProcessingContext,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

IMAGE_TOKEN: str
VIDEO_TOKEN: str

def get_num_combined_frames(
    num_frames: int, max_grid_shape: tuple[int, int] = (3, 3)
) -> int: ...

class HCXVisionImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values_images: Annotated[list[torch.Tensor], None]
    image_sizes_images: Annotated[torch.Tensor, None]

HCXVisionImageInputs = HCXVisionImagePixelInputs

class HCXVisionVideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: Annotated[list[list[torch.Tensor]], None]

HCXVisionVideoInputs = HCXVisionVideoPixelInputs

class HCXVisionProcessingInfo(BaseProcessingInfo):
    def get_vision_encoder_info(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(self, *, vision_query_length: int | list[int]) -> int: ...
    def get_num_video_tokens(self, *, vision_query_length: int | list[int]) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_max_image_tokens(self) -> int: ...

class HCXVisionDummyInputsBuilder(BaseDummyInputsBuilder[HCXVisionProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class HCXVisionMultiModalProcessor(
    BaseMultiModalProcessor[HCXVisionProcessingInfo]
): ...

def init_vision_tower_for_hcxvision(
    vision_config,
    quant_config: QuantizationConfig | None,
    *,
    use_nth_layer: int | None = None,
    require_post_norm: bool | None = None,
    prefix: str = "",
) -> CLIPVisionModel | SiglipVisionModel: ...

class HCXVisionMlp(nn.Module):
    mm_projector_type: Incomplete
    fc1: Incomplete
    act: Incomplete
    fc2: Incomplete
    def __init__(
        self,
        mm_projector_type,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=...,
    ) -> None: ...
    def forward(self, x): ...

class HCXVisionCAbstractor(nn.Module):
    num_input_tokens: Incomplete
    output_hidden_size: Incomplete
    pos_emb: Incomplete
    prenorm: Incomplete
    dtype: Incomplete
    def __init__(
        self,
        num_queries: int,
        num_input_tokens: int,
        encoder_hidden_size: int,
        hidden_size: int,
        output_hidden_size: int,
        pos_emb: bool = True,
        prenorm: bool = False,
    ) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: list[list[int]] | None = None,
        num_grids: list[int] | None = None,
    ) -> torch.Tensor: ...
    net: Incomplete
    readout: Incomplete
    def build_net(
        self,
        n_queries: int,
        encoder_hidden_size: int,
        hidden_size: int,
        output_hidden_size: int,
        depth: int = 3,
        mlp_depth: int = 2,
    ): ...
    def build_mlp(self, depth: int, hidden_size: int, output_hidden_size: int): ...

class HCXVisionForCausalLM(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    dtype: Incomplete
    vision_model: Incomplete
    mm_projector: Incomplete
    image_newline: Incomplete
    language_model: Incomplete
    config: Incomplete
    vision_config: Incomplete
    text_config: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def forward_images(
        self, pixel_values_images: list[torch.Tensor], image_sizes_images: torch.Tensor
    ) -> tuple[torch.Tensor, ...]: ...
    def forward_videos(
        self, pixel_values_videos: list[list[torch.Tensor]]
    ) -> tuple[torch.Tensor, ...]: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

def unpad_image(
    tensor: torch.Tensor, original_size: tuple[int, int]
) -> torch.Tensor: ...
def select_best_resolution(
    original_size: tuple, possible_resolutions: list
) -> tuple: ...
def get_anyres_image_grid_shape(
    image_size: tuple[int, int],
    grid_pinpoints: str | list[tuple[int, int]],
    patch_size: int,
) -> tuple[int, int]: ...
def reshape_and_unpad_image_features(
    image_feature: torch.Tensor,
    height: int,
    width: int,
    image_size: tuple[int, int],
    possible_resolutions: list[tuple[int, int]],
    grid_size: int,
    unpad: bool,
    image_newline: torch.Tensor,
) -> torch.Tensor: ...
def anyres_postprocessing(
    image_forward_outs: list[torch.Tensor],
    image_sizes: list[list[int]],
    possible_resolutions: list[tuple[int, int]],
    patch_size: int,
    grid_size: int,
    image_newline: torch.Tensor,
    num_queries_vis_abstractor: int = -1,
    unpad: bool = False,
) -> list[torch.Tensor]: ...
