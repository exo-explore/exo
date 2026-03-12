import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import (
    get_vit_attn_backend as get_vit_attn_backend,
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model as run_dp_sharded_mrope_vision_model,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Iterator, Mapping
from transformers import BatchFeature as BatchFeature
from transformers.models.qwen2_vl import Qwen2VLImageProcessor, Qwen2VLProcessor
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig,
    Qwen2VLVisionConfig as Qwen2VLVisionConfig,
)
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    parallel_state as parallel_state,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import QuickGELU as QuickGELU
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv3dLayer as Conv3dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb as ApplyRotaryEmb,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem as ImageItem,
    ModalityData as ModalityData,
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
    VideoItem as VideoItem,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems as DictEmbeddingItems,
    ImageSize as ImageSize,
    ModalityDataItems as ModalityDataItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)

logger: Incomplete

class Qwen2VLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

class Qwen2VLImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    image_embeds: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

Qwen2VLImageInputs: TypeAlias = Qwen2VLImagePixelInputs | Qwen2VLImageEmbeddingInputs

class Qwen2VLVideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: Annotated[torch.Tensor, None]
    video_grid_thw: Annotated[torch.Tensor, None]

class Qwen2VLVideoEmbeddingInputs(TensorSchema):
    type: Literal["video_embeds"]
    video_embeds: Annotated[torch.Tensor, None]
    video_grid_thw: Annotated[torch.Tensor, None]

Qwen2VLVideoInputs: TypeAlias = Qwen2VLVideoPixelInputs | Qwen2VLVideoEmbeddingInputs

class Qwen2VisionMLP(nn.Module):
    fc1: Incomplete
    act: Incomplete
    fc2: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: type[nn.Module] = ...,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Qwen2VisionAttention(nn.Module):
    tp_size: Incomplete
    tp_rank: Incomplete
    hidden_size_per_attention_head: Incomplete
    num_attention_heads_per_partition: Incomplete
    qkv: Incomplete
    proj: Incomplete
    attn: Incomplete
    apply_rotary_emb: Incomplete
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]: ...
    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor: ...

class Qwen2VisionBlock(nn.Module):
    norm1: Incomplete
    norm2: Incomplete
    attn: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: type[nn.Module] = ...,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor: ...

class Qwen2VisionPatchEmbed(nn.Module):
    patch_size: Incomplete
    temporal_patch_size: Incomplete
    embed_dim: Incomplete
    proj: Incomplete
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Qwen2VisionPatchMerger(nn.Module):
    hidden_size: Incomplete
    ln_q: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Qwen2VisionTransformer(nn.Module):
    use_data_parallel: Incomplete
    out_hidden_size: Incomplete
    spatial_merge_size: Incomplete
    num_heads: Incomplete
    embed_dim: Incomplete
    patch_embed: Incomplete
    rotary_pos_emb: Incomplete
    blocks: Incomplete
    merger: Incomplete
    attn_backend: Incomplete
    def __init__(
        self,
        vision_config: Qwen2VLVisionConfig,
        norm_eps: float = 1e-06,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def rot_pos_emb(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> int | None: ...
    def forward(
        self, x: torch.Tensor, grid_thw: torch.Tensor | list[list[int]]
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen2VLMultiModalDataParser(MultiModalDataParser):
    def __init__(self, spatial_merge_size: int, *args, **kwargs) -> None: ...

class Qwen2VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> Qwen2VLProcessor: ...
    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessor: ...
    def get_data_parser(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]: ...
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: Qwen2VLImageProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...
    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor: Qwen2VLImageProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...
    def get_image_size_with_most_features(
        self, max_pixels: int | None = None
    ) -> ImageSize: ...
    def get_max_image_tokens(self) -> int: ...
    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        max_frames_per_video: int = ...,
    ) -> int: ...
    def get_max_video_tokens(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...

class Qwen2VLDummyInputsBuilder(BaseDummyInputsBuilder[Qwen2VLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Qwen2VLMultiModalProcessor(BaseMultiModalProcessor[Qwen2VLProcessingInfo]): ...

class Qwen2VLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsMRoPE,
    metaclass=abc.ABCMeta,
):
    hf_to_vllm_mapper: Incomplete
    supports_encoder_tp_data: bool
    def iter_mm_grid_thw(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, int, int, int, float]]: ...
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    use_data_parallel: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    visual: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
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
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int: ...
    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int: ...

class Tarsier2MultiModalProcessor(Qwen2VLMultiModalProcessor): ...

class Tarsier2ImageProcessor(Qwen2VLImageProcessor):
    def __init__(self, size: dict[str, int] | None = None, **kwargs) -> None: ...

class Tarsier2Processor(Qwen2VLProcessor):
    def __init__(
        self,
        image_processor: Tarsier2ImageProcessor,
        tokenizer: TokenizerLike,
        video_processor: Qwen2VLVideoProcessor,
        **kwargs,
    ) -> None: ...

class Tarsier2ProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self) -> Qwen2VLConfig: ...
    def get_hf_processor(self, **kwargs: object) -> Tarsier2Processor: ...
    def get_image_processor(self) -> Tarsier2ImageProcessor: ...

class Tarsier2ForConditionalGeneration(
    Qwen2VLForConditionalGeneration, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
