import abc
import torch
import torch.nn as nn
from .vision import get_vit_attn_backend as get_vit_attn_backend
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Iterator
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import parallel_state as parallel_state
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention as MMEncoderAttention,
)
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
from vllm.model_executor.layers.quantization.gptq import GPTQConfig as GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig as GPTQMarlinConfig,
)
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb as ApplyRotaryEmb,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder as Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLMultiModalProcessor as Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo as Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems as MultiModalDataItems
from vllm.multimodal.processing import (
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
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

class OpenPanguVisionAttention(nn.Module):
    hidden_size_per_attention_head: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
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
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor: ...

class OpenPanguVisionMLP(nn.Module):
    hidden_act: Incomplete
    gate_up_proj: Incomplete
    up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = ...,
        vision_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor): ...

class OpenPanguVisionBlock(nn.Module):
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
        vision_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor: ...

class OpenPanguVisionRotaryEmbedding(nn.Module):
    inv_freq: Incomplete
    def __init__(self, dim: int, theta: float = 10000.0) -> None: ...
    def update_freqs_cache(self, seqlen: int) -> None: ...
    def forward(self, seqlen: int) -> torch.Tensor: ...

class OpenPanguVisionPatchEmbed(nn.Module):
    patch_size: Incomplete
    temporal_patch_size: Incomplete
    hidden_size: Incomplete
    input_size: Incomplete
    proj: Incomplete
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class OpenPanguVisionPatchMerger(nn.Module):
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

class OpenPanguVisionTransformer(nn.Module):
    hidden_size: Incomplete
    num_heads: Incomplete
    window_size: Incomplete
    patch_size: Incomplete
    spatial_merge_size: Incomplete
    fullatt_block_indexes: Incomplete
    spatial_merge_unit: Incomplete
    interleaved: Incomplete
    out_hidden_size: Incomplete
    hidden_act: Incomplete
    attn_backend: Incomplete
    rotary_pos_emb: Incomplete
    patch_embed: Incomplete
    blocks: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
    hidden_size_per_attention_head: Incomplete
    select_layer: Incomplete
    select_index: Incomplete
    take_indices: Incomplete
    final_layernorm: Incomplete
    merger: Incomplete
    vision_projection: Incomplete
    def __init__(
        self,
        vision_config,
        out_hidden_size,
        hidden_size,
        norm_eps: float = 1e-06,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        interleaved: bool = False,
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def cal_cos_sin(self, rotary_pos_emb): ...
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor: ...
    def get_window_index(self, grid_thw): ...
    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights) -> set[str]: ...

class ProjectionSingle(nn.Module):
    act: Incomplete
    fc1: Incomplete
    def __init__(self, i_hidden_size: int, t_hidden_size: int) -> None: ...
    def forward(self, hidden_states): ...

class OpenPanguVLProcessingInfo(Qwen2_5_VLProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(
        self,
        *,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        size: dict[str, int] | None = None,
        fps: float | list[float] | None = None,
        **kwargs: object,
    ): ...

class OpenPanguVLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

class OpenPanguVLImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    image_embeds: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

class OpenPanguVLVideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: Annotated[torch.Tensor, None]
    video_grid_thw: Annotated[torch.Tensor, None]

class OpenPanguVLVideoEmbeddingInputs(TensorSchema):
    type: Literal["video_embeds"]
    video_embeds: Annotated[torch.Tensor, None]
    video_grid_thw: Annotated[torch.Tensor, None]

class OpenPanguVLMultiModalProcessor(Qwen2_5_VLMultiModalProcessor): ...
class OpenPanguVLDummyInputsBuilder(Qwen2_5_VLDummyInputsBuilder): ...

class OpenPanguVLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsMRoPE,
    metaclass=abc.ABCMeta,
):
    hf_to_vllm_mapper: Incomplete
    packed_modules_mapping: Incomplete
    config: Incomplete
    vllm_config: Incomplete
    visual: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None: ...
    def get_input_embeddings(
        self, input_ids: torch.Tensor, multimodal_embeddings=None
    ) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata=None
    ) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    def iter_mm_grid_thw(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[str, int, int, int, int]]: ...
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...

def rescale(image, scale): ...
def normalize(image, mean, std): ...
def rescale_and_normalize(
    images: torch.Tensor,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: float | list[float],
    image_std: float | list[float],
    dtype: torch.dtype = ...,
) -> torch.Tensor: ...
