import abc
import torch
import torch.nn as nn
from .ernie45_vl_moe import Ernie4_5_VLMoeForCausalLM as Ernie4_5_VLMoeForCausalLM
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
    maybe_prefix as maybe_prefix,
)
from .vision import get_vit_attn_backend as get_vit_attn_backend
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Mapping
from transformers import BaseImageProcessor as BaseImageProcessor
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import (
    BaseDummyOptions as BaseDummyOptions,
    VideoDummyOptions as VideoDummyOptions,
)
from vllm.distributed import parallel_state as parallel_state
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import QuickGELU as QuickGELU
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb as ApplyRotaryEmb,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize as ImageSize,
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
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)

logger: Incomplete

def all_gather_interleave(local_tensor, hidden_size: int, tp_size: int): ...

class Ernie4_5_VisionAttention(nn.Module):
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
        rotary_pos_emb: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class Ernie4_5_VisionMLP(nn.Module):
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

class Ernie4_5_VisionBlock(nn.Module):
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
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class Ernie4_5_VisionPatchEmbed(nn.Module):
    patch_size: Incomplete
    in_channels: Incomplete
    embed_dim: Incomplete
    proj: Incomplete
    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1280,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Ernie4_5_VisionRotaryEmbedding(nn.Module):
    inv_freq: Incomplete
    def __init__(self, dim: int, theta: float = 10000.0) -> None: ...
    def forward(self, seqlen: int) -> torch.Tensor: ...

class Ernie4_5_VisionTransformer(nn.Module):
    spatial_merge_size: Incomplete
    num_heads: Incomplete
    embed_dim: Incomplete
    patch_embed: Incomplete
    rotary_pos_emb: Incomplete
    blocks: Incomplete
    ln: Incomplete
    attn_backend: Incomplete
    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-06,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor: ...
    def compute_attn_mask_seqlen(
        self, cu_seqlens: torch.Tensor
    ) -> torch.Tensor | None: ...
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, num_pad: int = 0
    ) -> torch.Tensor: ...
    def load_weights(self, weights) -> set[str]: ...

class Ernie4_5_VLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

Ernie4_5_VLImageInputs = Ernie4_5_VLImagePixelInputs

class Ernie4_5_VLVideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: Annotated[torch.Tensor, None]
    video_grid_thw: Annotated[torch.Tensor, None]

Ernie4_5_VLVideoInputs = Ernie4_5_VLVideoPixelInputs

def round_by_factor(number: int | float, factor: int) -> int: ...
def ceil_by_factor(number: int | float, factor: int) -> int: ...
def floor_by_factor(number: int | float, factor: int) -> int: ...
def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = ...,
    max_pixels: int = ...,
): ...

class VariableResolutionResamplerModel(nn.Module):
    in_dim: Incomplete
    out_dim: Incomplete
    config: Incomplete
    spatial_conv_size: Incomplete
    temporal_conv_size: Incomplete
    use_temporal_conv: Incomplete
    spatial_dim: Incomplete
    temporal_dim: Incomplete
    spatial_linear1: Incomplete
    spatial_gelu: Incomplete
    spatial_linear2: Incomplete
    spatial_norm: Incomplete
    temporal_linear1: Incomplete
    temporal_gelu: Incomplete
    temporal_linear2: Incomplete
    temporal_norm: Incomplete
    mlp: Incomplete
    after_norm: Incomplete
    def __init__(
        self,
        in_dim,
        out_dim,
        spatial_conv_size,
        temporal_conv_size,
        config,
        prefix: str = "",
    ) -> None: ...
    def spatial_conv_reshape(self, x, spatial_conv_size): ...
    def forward(self, x, grid_thw): ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Ernie4_5_VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_image_processor(self, **kwargs: object): ...
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
        image_processor: BaseImageProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...
    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor: BaseImageProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_max_image_tokens(self) -> int: ...
    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...
    def get_max_video_tokens(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...

class Ernie4_5VLMultiModalProcessor(
    BaseMultiModalProcessor[Ernie4_5_VLProcessingInfo]
): ...

class Ernie4_5_VLDummyInputsBuilder(BaseDummyInputsBuilder[Ernie4_5_VLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Ernie4_5_VLMoeForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsMRoPE,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    vision_model: Incomplete
    resampler_model: Incomplete
    language_model: Incomplete
    visual_token_mask: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None: ...
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
        **kwargs,
    ): ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
