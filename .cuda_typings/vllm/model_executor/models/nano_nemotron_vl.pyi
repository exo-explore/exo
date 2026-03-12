import abc
import numpy.typing as npt
import torch
import torch.nn as nn
from PIL import Image
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property as cached_property
from transformers import (
    BatchFeature,
    PretrainedConfig as PretrainedConfig,
    TensorType as TensorType,
)
from typing import Annotated, Any, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import (
    BaseDummyOptions as BaseDummyOptions,
    VideoDummyOptions as VideoDummyOptions,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import (
    ReLUSquaredActivation as ReLUSquaredActivation,
)
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState as HasInnerState,
    IsHybrid as IsHybrid,
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsMultiModalPruning as SupportsMultiModalPruning,
)
from vllm.model_executor.models.internvl import (
    calculate_internvl_targets as calculate_internvl_targets,
    get_internvl_target_ratios as get_internvl_target_ratios,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.nemotron_h import (
    NemotronHForCausalLM as NemotronHForCausalLM,
)
from vllm.model_executor.models.parakeet import (
    ParakeetExtractor as ParakeetExtractor,
    ProjectedParakeet as ProjectedParakeet,
)
from vllm.model_executor.models.radio import (
    RadioModel as RadioModel,
    calc_seq_lens as calc_seq_lens,
)
from vllm.model_executor.models.utils import (
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.evs import (
    compute_retained_tokens_count as compute_retained_tokens_count,
    compute_retention_mask as compute_retention_mask,
)
from vllm.multimodal.inputs import (
    AudioItem as AudioItem,
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalInputs as MultiModalInputs,
    MultiModalKwargsItems as MultiModalKwargsItems,
    VideoItem as VideoItem,
)
from vllm.multimodal.media.audio import (
    extract_audio_from_video_bytes as extract_audio_from_video_bytes,
)
from vllm.multimodal.parse import (
    AudioProcessorItems as AudioProcessorItems,
    ImageEmbeddingItems as ImageEmbeddingItems,
    ImageProcessorItems as ImageProcessorItems,
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
    VideoProcessorItems as VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    ProcessorInputs as ProcessorInputs,
    TimingContext as TimingContext,
)
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import (
    TokenizerLike as TokenizerLike,
    cached_tokenizer_from_config as cached_tokenizer_from_config,
)
from vllm.transformers_utils.configs.radio import RadioConfig as RadioConfig
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete

class NanoNemotronVLAudioFeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    input_audio_features: Annotated[torch.Tensor, None]
    feature_attention_mask: Annotated[torch.Tensor, None]
    audio_feature_lengths: Annotated[torch.Tensor, None]

MAX_AUDIO_LEN_S: Incomplete
IMG_START: str
IMG_END: str
IMG_CONTEXT: str
AUDIO_START: str
AUDIO_END: str
AUDIO_CONTEXT: str
DEFAULT_NUM_TILES: int

class NanoNemotronVLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values_flat: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

class NanoNemotronVLImagePixelInputsDynamic(TensorSchema):
    type: Literal["pixel_values_dynamic"]
    pixel_values_flat: Annotated[torch.Tensor, None]
    imgs_sizes: list[tuple[int, int]]
    num_tokens_per_image: list[int]

class NanoNemotronVLImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], None]

NanoNemotronVLImageInputs: TypeAlias = (
    NanoNemotronVLImagePixelInputs
    | NanoNemotronVLImagePixelInputsDynamic
    | NanoNemotronVLImageEmbeddingInputs
)

class NanoNemotronVLVideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]
    pixel_values_flat: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]
    frames_indices: Annotated[torch.Tensor, None]
    frame_duration_ms: Annotated[torch.Tensor, None]

class NanoNemotronVLVideoEmbeddingInputs(TensorSchema):
    type: Literal["video_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], None]

NanoNemotronVLVideoInputs: TypeAlias = (
    NanoNemotronVLVideoPixelInputs | NanoNemotronVLVideoEmbeddingInputs
)

def dynamic_preprocess(
    image,
    *,
    image_size: int = 512,
    max_num_tiles: int = 12,
    use_thumbnail: bool = True,
    idx: int = 0,
): ...
def image_to_pixel_values(
    image: Image.Image, *, input_size: int, max_num: int, use_thumbnail: bool, idx: int
) -> torch.Tensor: ...
def video_to_pixel_values(
    video: npt.NDArray, *, input_size: int, max_num_tiles: int = 1, use_thumbnail: bool
) -> torch.Tensor: ...
def input_conditioner(x, norm_mean, norm_std): ...
def calculate_timestamps(indices: list[int] | torch.Tensor, frame_duration_ms: int): ...

class DynamicResolutionImageTiler:
    CONV_MERGING: bool
    PIXEL_SHUFFLE: bool
    USE_THUMBNAIL: bool
    norm_mean: Incomplete
    norm_std: Incomplete
    def __init__(
        self,
        *,
        max_model_len: int,
        patch_size: int,
        min_num_patches: int,
        max_num_patches: int,
        downsample_ratio: int,
        norm_mean: Sequence[float],
        norm_std: Sequence[float],
        factor_max: float = 1.0,
        use_thumbnail: bool = False,
    ) -> None: ...
    def width_and_height_for_max_num_tokens_available(
        self, target_num_tokens_post_shuffle: int
    ) -> tuple[int, int]: ...
    def max_num_tokens_available(self, text_prompt_length: int) -> int: ...
    feature_size_cache: dict[Image.Image, int]
    @classmethod
    def get_cached_feature_size(cls, image: Image.Image) -> int: ...
    @dataclass
    class DynamicResolutionParams:
        media: Image.Image
        num_tiles: int
        num_embeddings: int
        patch_size: tuple[int, int]

    def apply_params(self, params: DynamicResolutionParams) -> list[torch.Tensor]: ...
    def process_media(
        self, media: Image.Image, num_tokens_available: int
    ) -> tuple[DynamicResolutionParams, int]: ...
    def compute_params(
        self, media_list: list[Image.Image], num_tokens_available: int | None = None
    ) -> list[DynamicResolutionParams]: ...
    @staticmethod
    def stack(images: list[torch.Tensor], patch_size: int) -> torch.Tensor: ...

class BaseNanoNemotronVLProcessor(ABC, metaclass=abc.ABCMeta):
    config: Incomplete
    tokenizer: Incomplete
    max_num_tiles: Incomplete
    num_image_token: Incomplete
    image_size: Incomplete
    use_thumbnail: bool
    norm_mean: Incomplete
    norm_std: Incomplete
    dynamic_tiler: DynamicResolutionImageTiler | None
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *args,
        max_model_len: int,
        max_num_tiles: int | None = None,
        **kwargs,
    ) -> None: ...
    @staticmethod
    def use_dynamic_resolution(config: PretrainedConfig) -> bool: ...
    @property
    @abstractmethod
    def image_token_id(self) -> int: ...
    @abstractmethod
    def get_image_repl(
        self, feature_size: int, num_patches: int | None
    ) -> PromptUpdateDetails[str]: ...
    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, max_num_tiles: int
    ) -> int: ...
    @abstractmethod
    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        return_tensors: str | TensorType | None = None,
        max_num_tiles: int | None = None,
    ) -> BatchFeature: ...

class NanoNemotronVLProcessor(BaseNanoNemotronVLProcessor):
    video_token: Incomplete
    video_pruning_rate: Incomplete
    audio_extractor: ParakeetExtractor | None
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *,
        max_model_len: int,
        max_num_tiles: int | None = None,
        video_token: str | None = None,
        video_pruning_rate: float | None = None,
    ) -> None: ...
    @property
    def supports_video(self) -> bool: ...
    @property
    def video_token_id(self) -> int | None: ...
    @property
    def image_token_id(self) -> int: ...
    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        videos: list[tuple[npt.NDArray, dict[str, Any]]] | None = None,
        audios: AudioItem | list[AudioItem] | None = None,
        return_tensors: str | TensorType | None = None,
        max_num_tiles: int | None = None,
    ) -> BatchFeature: ...
    def get_image_repl(
        self, feature_size: int, num_patches: int | None
    ) -> PromptUpdateDetails[str]: ...
    def get_audio_repl(self, audio: npt.NDArray) -> PromptUpdateDetails[str]: ...
    @classmethod
    def get_video_repl(
        cls,
        *,
        tokens_per_frame: list[int],
        frames_indices: list[int],
        frame_duration_ms: int,
        tokenizer: TokenizerLike,
        img_start_token_ids: list[int],
        img_end_token_ids: list[int],
        img_context_token_ids: list[int],
    ) -> PromptUpdateDetails[list[int]]: ...

class BaseNanoNemotronVLProcessingInfo(BaseProcessingInfo, metaclass=abc.ABCMeta):
    @abstractmethod
    def get_hf_processor(self, **kwargs: object) -> BaseNanoNemotronVLProcessor: ...
    def get_default_tok_params(self) -> TokenizeParams: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_image_size_with_most_features(self, max_num_tiles: int) -> ImageSize: ...
    def get_max_image_tokens(self) -> int: ...

class NanoNemotronVLProcessingInfo(BaseNanoNemotronVLProcessingInfo):
    @property
    def supports_video(self): ...
    @property
    def audio_extractor(self) -> ParakeetExtractor | None: ...
    def get_data_parser(self): ...
    def get_supported_mm_limits(self): ...
    def get_video_token(self) -> str | None: ...
    def get_video_pruning_rate(self) -> float | None: ...
    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...
    def get_hf_processor(self, **kwargs: object) -> NanoNemotronVLProcessor: ...

class NanoNemotronBaseVLMultiModalProcessor(BaseMultiModalProcessor[_I]):
    @cached_property
    def is_dynamic_tiler(self) -> bool: ...

class NanoNemotronVLMultiModalProcessor(
    NanoNemotronBaseVLMultiModalProcessor[NanoNemotronVLProcessingInfo]
):
    def apply(
        self, processor_inputs: ProcessorInputs, timing_ctx: TimingContext | None = None
    ) -> MultiModalInputs: ...

class NanoNemotronVLDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class NanoNemotronVLDummyInputsBuilder(
    NanoNemotronVLDummyInputsBuilder[NanoNemotronVLProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class NemotronH_Nano_VL_V2(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsMultiModal,
    SupportsMultiModalPruning,
    metaclass=abc.ABCMeta,
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    patch_size: Incomplete
    template: Incomplete
    num_image_token: Incomplete
    downsample_ratio: Incomplete
    ps_version: Incomplete
    image_tag_type: Incomplete
    video_pruning_rate: Incomplete
    language_model: Incomplete
    llm_dtype: Incomplete
    vision_model: Incomplete
    mlp1: Incomplete
    sound_encoder: ProjectedParakeet | None
    config: Incomplete
    model_config: Incomplete
    dynamic_resolution: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def pixel_shuffle(self, x, scale_factor: float = 0.5): ...
    def pixel_shuffle_dynamic_res(
        self, x: torch.Tensor, *, imgs_sizes: list[tuple[int, int]]
    ) -> torch.Tensor: ...
    def extract_feature_dynamic(
        self, pixel_values: torch.Tensor, imgs_sizes: list[tuple[int, int]]
    ): ...
    def extract_feature(self, pixel_values: torch.Tensor): ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
    def get_vit_model_from_radio_config(self, hf_config): ...
    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs): ...
    def get_seqlen_agnostic_capture_inputs(self, batch_size: int): ...
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig): ...
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig): ...
    @classmethod
    def get_mamba_state_copy_func(cls): ...
