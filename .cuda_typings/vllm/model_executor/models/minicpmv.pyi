import abc
import numpy as np
import torch
from .idefics2_vision_model import (
    Idefics2VisionTransformer as Idefics2VisionTransformer,
)
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    flatten_bn as flatten_bn,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Mapping
from torch import nn
from transformers import PretrainedConfig as PretrainedConfig
from typing import Annotated, Any, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.resampler import (
    BaseResampler as BaseResampler,
    Resampler2 as Resampler2,
    get_2d_sincos_pos_embed as get_2d_sincos_pos_embed,
)
from vllm.model_executor.models.llama import LlamaForCausalLM as LlamaForCausalLM
from vllm.model_executor.models.minicpm import MiniCPMForCausalLM as MiniCPMForCausalLM
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM as Qwen2ForCausalLM
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM as Qwen3ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
    NestedTensors as NestedTensors,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems as DictEmbeddingItems,
    ImageItem as ImageItem,
    ImageProcessorItems as ImageProcessorItems,
    ImageSize as ImageSize,
    ModalityData as ModalityData,
    ModalityDataItems as ModalityDataItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
    VideoItem as VideoItem,
    VideoProcessorItems as VideoProcessorItems,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder as BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
    ResolvedPromptUpdate as ResolvedPromptUpdate,
)
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.collection_utils import flatten_2d_lists as flatten_2d_lists
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)
from vllm.utils.torch_utils import set_default_torch_dtype as set_default_torch_dtype

class MiniCPMVImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[list[torch.Tensor], None]
    tgt_sizes: Annotated[torch.Tensor, None]
    num_slices: Annotated[torch.Tensor, None]

class MiniCPMVImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    image_embeds: Annotated[torch.Tensor | list[torch.Tensor], None]

MiniCPMVImageInputs: TypeAlias = MiniCPMVImagePixelInputs | MiniCPMVImageEmbeddingInputs
DEFAULT_LN: Incomplete

class Resampler2_5(BaseResampler):
    max_size: Incomplete
    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: int | None = None,
        norm_layer: Callable[[int], nn.LayerNorm] = ...,
        max_size: tuple[int, int] = (70, 70),
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor, tgt_sizes: torch.Tensor) -> torch.Tensor: ...

class Resampler4_5(Resampler2_5):
    max_temporal_size: Incomplete
    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: int | None = None,
        norm_layer: Callable[[int], nn.LayerNorm] = ...,
        max_size: tuple[int, int] = (70, 70),
        max_temporal_size: int = 36000,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def get_1d_sincos_pos_embed_from_temporal_size(
        self, embed_dim: int, pos: np.ndarray
    ): ...
    def forward(
        self, x: torch.Tensor, tgt_sizes: torch.Tensor, temporal_ids=None
    ) -> torch.Tensor: ...

def get_version_by_config(config: PretrainedConfig) -> tuple[int, ...]: ...

class MiniCPMVImageEmbeddingItems(DictEmbeddingItems):
    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]], Mapping[str, MultiModalFieldConfig]
        ],
    ) -> None: ...
    def get_image_size(self, index: int) -> ImageSize: ...

class MiniCPMVVideoEmbeddingItems(DictEmbeddingItems):
    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]], Mapping[str, MultiModalFieldConfig]
        ],
    ) -> None: ...
    def get_frame_size(self, index: int) -> ImageSize: ...
    def get_num_frames(self, index: int) -> int: ...

class MiniCPMVMultiModalDataParser(MultiModalDataParser): ...

class MiniCPMVProcessingInfo(BaseProcessingInfo):
    image_pattern: str
    video_pattern: str
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_image_processor(self, **kwargs: object): ...
    def get_data_parser(self): ...
    def get_model_version(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_slice_image_placeholder(
        self,
        image_size: ImageSize,
        image_idx: int = 0,
        max_slice_nums: int | None = None,
        use_image_id: bool = True,
    ) -> str: ...
    def get_sliced_grid(
        self, image_size: ImageSize, max_slice_nums: int | None = None
    ) -> tuple[int, int] | None: ...
    def get_num_image_tokens(
        self, image_size: ImageSize, max_slice_nums: int | None = None
    ) -> int: ...
    def get_max_image_tokens(self) -> int: ...
    def get_image_max_slice_num(self) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_max_video_frame_tokens(self) -> int: ...
    def get_max_video_tokens(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...
    def get_video_max_slice_num(self) -> int: ...
    def get_video_frame_size_with_most_features(self) -> ImageSize: ...
    def get_max_video_frames(self, max_tokens: int) -> int: ...
    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...

class MiniCPMVDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class MiniCPMVMultiModalProcessor(BaseMultiModalProcessor[_I]):
    def get_image_prompt_texts(
        self, image_size: ImageSize, image_idx: int = 0
    ) -> str: ...
    def get_video_prompt_texts(self, image_size: ImageSize, num_frames: int) -> str: ...
    def process_images(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]: ...
    def process_videos(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]: ...
    def process_mm_inputs(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]: ...

class MiniCPMVBaseModel(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    supports_encoder_tp_data: bool
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    use_data_parallel: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    version: Incomplete
    llm: Incomplete
    vpm: Incomplete
    vision_dim: Incomplete
    embed_dim: Incomplete
    resampler: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def init_llm(self, vllm_config: VllmConfig, prefix: str = "") -> nn.Module: ...
    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> nn.Module: ...
    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module: ...
    def get_vision_hidden_states(
        self, data: MiniCPMVImagePixelInputs
    ) -> torch.Tensor: ...

class MiniCPMV2_0(MiniCPMVBaseModel, metaclass=abc.ABCMeta):
    supports_encoder_tp_data: bool
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def init_llm(self, vllm_config: VllmConfig, prefix: str = "") -> nn.Module: ...
    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> nn.Module: ...
    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module: ...
    def get_vision_hidden_states(
        self, data: MiniCPMVImagePixelInputs
    ) -> torch.Tensor: ...

class MiniCPMV2_5(MiniCPMVBaseModel, SupportsLoRA, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def init_llm(self, vllm_config: VllmConfig, prefix: str = "") -> nn.Module: ...
    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> nn.Module: ...
    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module: ...
    def get_vision_hidden_states(
        self, data: MiniCPMVImagePixelInputs
    ) -> torch.Tensor: ...

class MiniCPMV2_6(MiniCPMVBaseModel, SupportsLoRA, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def init_llm(self, vllm_config: VllmConfig, prefix: str = "") -> nn.Module: ...
    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module: ...
    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module: ...
    def get_vision_hidden_states(
        self, data: MiniCPMVImagePixelInputs
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class MiniCPMV4_0(MiniCPMVBaseModel, SupportsLoRA, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def init_llm(self, vllm_config: VllmConfig, prefix: str = "") -> nn.Module: ...
    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module: ...
    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module: ...
    def get_vision_hidden_states(
        self, data: MiniCPMVImagePixelInputs
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class MiniCPMV4_5(MiniCPMVBaseModel, SupportsLoRA, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def init_llm(self, vllm_config: VllmConfig, prefix: str = "") -> nn.Module: ...
    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module: ...
    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module: ...
    def get_vision_hidden_states(
        self, data: MiniCPMVImagePixelInputs
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class MiniCPMV(
    MiniCPMVBaseModel, SupportsMultiModal, SupportsLoRA, metaclass=abc.ABCMeta
):
    def __new__(cls, *, vllm_config: VllmConfig, prefix: str = ""): ...
