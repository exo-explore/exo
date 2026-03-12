import abc
import torch
import torch.nn as nn
from .vision import (
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model as run_dp_sharded_mrope_vision_model,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers.models.qwen2_vl import Qwen2VLProcessor
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
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
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb as ApplyRotaryEmb,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM as Qwen2ForCausalLM
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLDummyInputsBuilder as Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalProcessor as Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo as Qwen2VLProcessingInfo,
    Qwen2VisionAttention as Qwen2VisionAttention,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from vllm.model_executor.models.vision import (
    get_vit_attn_backend as get_vit_attn_backend,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict as MultiModalDataDict
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs.dotsocr import (
    DotsOCRConfig as DotsOCRConfig,
    DotsVisionConfig as DotsVisionConfig,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)

IMAGE_TOKEN: str

class DotsOCRImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

class DotsOCRImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    image_embeds: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

DotsOCRImageInputs: TypeAlias = DotsOCRImagePixelInputs | DotsOCRImageEmbeddingInputs

class DotsOCRDummyInputsBuilder(Qwen2VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class DotsOCRProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self) -> DotsOCRConfig: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]: ...
    def get_hf_processor(self, **kwargs: object) -> Qwen2VLProcessor: ...

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None: ...
    def forward(self, seqlen: int) -> torch.Tensor: ...

class PatchMerger(nn.Module):
    hidden_size: Incomplete
    pre_norm: Incomplete
    ln_q: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        pre_norm: str = "layernorm",
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class DotsVisionAttention(nn.Module):
    embed_dim: Incomplete
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
        config,
        dim: int,
        num_heads: int = 16,
        bias: bool = True,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        *,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class DotsSwiGLUFFN(nn.Module):
    fc13: Incomplete
    fc2: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class DotsPatchEmbed(nn.Module):
    num_channels: Incomplete
    patch_size: Incomplete
    temporal_patch_size: Incomplete
    embed_dim: Incomplete
    config: Incomplete
    proj: Incomplete
    norm: Incomplete
    def __init__(self, config) -> None: ...
    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor: ...

class DotsViTPreprocessor(nn.Module):
    patch_h: Incomplete
    patch_w: Incomplete
    embed_dim: Incomplete
    config: Incomplete
    patchifier: Incomplete
    def __init__(self, config) -> None: ...
    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor: ...

class DotsVisionBlock(nn.Module):
    attn: Incomplete
    norm1: Incomplete
    mlp: Incomplete
    norm2: Incomplete
    def __init__(
        self,
        config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor: ...

class DotsVisionTransformer(nn.Module):
    config: Incomplete
    spatial_merge_size: Incomplete
    patch_embed: Incomplete
    rotary_pos_emb: Incomplete
    attn_backend: Incomplete
    out_hidden_size: Incomplete
    blocks: Incomplete
    post_trunk_norm: Incomplete
    merger: Incomplete
    def __init__(
        self,
        config: DotsVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def get_pos_ids_by_grid(self, grid_thw: list[list[int]]) -> list[torch.Tensor]: ...
    def rot_pos_emb(self, grid_thw: list[list[int]]) -> torch.Tensor: ...
    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> int | None: ...
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor: ...

class DotsOCRForCausalLM(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    packed_modules_mapping: Incomplete
    supports_encoder_tp_data: bool
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: DotsOCRConfig
    quant_config: Incomplete
    use_data_parallel: Incomplete
    vision_tower: Incomplete
    language_model: Qwen2ForCausalLM
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int: ...
    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
