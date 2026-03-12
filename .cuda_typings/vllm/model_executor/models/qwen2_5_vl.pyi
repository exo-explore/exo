import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsEagle3 as SupportsEagle3,
    SupportsLoRA as SupportsLoRA,
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
    SupportsMultiModalPruning as SupportsMultiModalPruning,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
)
from .qwen2_vl import (
    Qwen2VLMultiModalProcessor as Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo as Qwen2VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    cast_overflow_tensors as cast_overflow_tensors,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import (
    get_vit_attn_backend as get_vit_attn_backend,
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model as run_dp_sharded_mrope_vision_model,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Iterator
from transformers import BatchFeature as BatchFeature
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig as Qwen2_5_VLVisionConfig,
)
from typing import Annotated, Literal, TypeAlias
from vllm.compilation.decorators import (
    should_torch_compile_mm_encoder as should_torch_compile_mm_encoder,
    support_torch_compile as support_torch_compile,
)
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import parallel_state as parallel_state
from vllm.forward_context import set_forward_context as set_forward_context
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import (
    get_act_and_mul_fn as get_act_and_mul_fn,
)
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv3dLayer as Conv3dLayer
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
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
from vllm.multimodal.evs import (
    compute_mrope_for_media as compute_mrope_for_media,
    compute_retained_tokens_count as compute_retained_tokens_count,
    compute_retention_mask as compute_retention_mask,
    recompute_mrope_positions as recompute_mrope_positions,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems as MultiModalDataItems
from vllm.multimodal.processing import (
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)

logger: Incomplete

class Qwen2_5_VLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

class Qwen2_5_VLImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    image_embeds: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

Qwen2_5_VLImageInputs: TypeAlias = (
    Qwen2_5_VLImagePixelInputs | Qwen2_5_VLImageEmbeddingInputs
)

class Qwen2_5_VLVideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: Annotated[torch.Tensor, None]
    video_grid_thw: Annotated[torch.Tensor, None]
    second_per_grid_ts: Annotated[torch.Tensor | None, None]
    timestamps: list[list[float]] | None

class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):
    type: Literal["video_embeds"]
    video_embeds: Annotated[torch.Tensor, None]
    video_grid_thw: Annotated[torch.Tensor, None]
    second_per_grid_ts: Annotated[torch.Tensor | None, None]
    timestamps: list[list[float]] | None

Qwen2_5_VLVideoInputs: TypeAlias = (
    Qwen2_5_VLVideoPixelInputs | Qwen2_5_VLVideoEmbeddingInputs
)

class Qwen2_5_VisionMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = ...,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor): ...

class Qwen2_5_VisionAttention(nn.Module):
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
    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor: ...

class Qwen2_5_VisionBlock(nn.Module):
    norm1: Incomplete
    norm2: Incomplete
    attn: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = ...,
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
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor: ...

class Qwen2_5_VisionPatchEmbed(nn.Module):
    patch_size: Incomplete
    temporal_patch_size: Incomplete
    hidden_size: Incomplete
    proj: Incomplete
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Qwen2_5_VisionPatchMerger(nn.Module):
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

class Qwen2_5_VisionTransformer(nn.Module):
    hidden_size: Incomplete
    num_heads: Incomplete
    out_hidden_size: Incomplete
    window_size: Incomplete
    patch_size: Incomplete
    spatial_merge_size: Incomplete
    fullatt_block_indexes: Incomplete
    spatial_merge_unit: Incomplete
    patch_embed: Incomplete
    rotary_pos_emb: Incomplete
    attn_backend: Incomplete
    blocks: Incomplete
    merger: Incomplete
    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-06,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def rotary_pos_emb_thw(self, t, h, w): ...
    def get_window_index_thw(self, grid_t, grid_h, grid_w): ...
    def get_rope_by_thw(self, t, h, w): ...
    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> torch.Tensor: ...
    @staticmethod
    def invert_permutation(perm: torch.Tensor) -> torch.Tensor: ...
    def forward(self, x: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen2_5_VLProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> Qwen2_5_VLProcessor: ...

class Qwen2_5_VLMultiModalProcessor(Qwen2VLMultiModalProcessor): ...

class Qwen2_5_VLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
    SupportsEagle3,
    SupportsMultiModalPruning,
    SupportsMRoPE,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
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
    vllm_config: Incomplete
    multimodal_config: Incomplete
    video_pruning_rate: Incomplete
    is_multimodal_pruning_enabled: Incomplete
    visual: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]: ...
    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: tuple[torch.Tensor, ...],
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, int]: ...
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
