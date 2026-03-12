import abc
import torch
import torch.nn as nn
from .interfaces import (
    SupportsLoRA as SupportsLoRA,
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .keye import (
    BaseKeyeModule as BaseKeyeModule,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    KeyeBaseDummyInputsBuilder as KeyeBaseDummyInputsBuilder,
    KeyeProcessingInfo as KeyeProcessingInfo,
)
from _typeshed import Incomplete
from collections.abc import Mapping
from transformers import PretrainedConfig as PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature as BatchFeature
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem as ImageItem,
    ModalityData as ModalityData,
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
    VideoItem as VideoItem,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems as DictEmbeddingItems,
    ModalityDataItems as ModalityDataItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import (
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete

def split_thw(grid_thw: torch.Tensor) -> torch.Tensor: ...
def get_num_patches(
    grid_thw: torch.Tensor, num_frames: list[int] | torch.Tensor
) -> list[int]: ...

class KeyeVL1_5ImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

class KeyeVL1_5ImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    image_embeds: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

KeyeVL1_5ImageInputs: TypeAlias = (
    KeyeVL1_5ImagePixelInputs | KeyeVL1_5ImageEmbeddingInputs
)

class KeyeVL1_5VideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: Annotated[torch.Tensor, None]
    video_grid_thw: Annotated[torch.Tensor, None]
    num_frames: torch.Tensor

class KeyeVL1_5VideoEmbeddingInputs(TensorSchema):
    type: Literal["video_embeds"]
    video_embeds: Annotated[torch.Tensor, None]
    video_grid_thw: Annotated[torch.Tensor, None]
    num_frames: torch.Tensor

KeyeVL1_5VideoInputs: TypeAlias = (
    KeyeVL1_5VideoPixelInputs | KeyeVL1_5VideoEmbeddingInputs
)

class KeyeVL1_5Projector(nn.Module):
    text_config: Incomplete
    vision_config: Incomplete
    merge_kernel_size: Incomplete
    hidden_size: Incomplete
    pre_norm: Incomplete
    act: Incomplete
    linear_1: Incomplete
    linear_2: Incomplete
    def __init__(
        self,
        text_config: PretrainedConfig,
        vision_config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        image_features: torch.Tensor | tuple[torch.Tensor] | list[torch.Tensor],
        image_grid_thw: list[tuple[int, int, int]],
    ) -> torch.Tensor | list[torch.Tensor]: ...

class KeyeVL1_5MultiModalDataParser(MultiModalDataParser): ...

class KeyeVL1_5ProcessingInfo(KeyeProcessingInfo):
    def get_data_parser(self): ...
    def get_max_frame_per_video(self) -> int: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...

class KeyeVL1_5MultiModalProcessor(
    BaseMultiModalProcessor[KeyeVL1_5ProcessingInfo]
): ...
class KeyeVL1_5DummyInputsBuilder(
    KeyeBaseDummyInputsBuilder[KeyeVL1_5ProcessingInfo]
): ...

class KeyeVL1_5ForConditionalGeneration(
    BaseKeyeModule,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsMRoPE,
    metaclass=abc.ABCMeta,
):
    merge_size: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
