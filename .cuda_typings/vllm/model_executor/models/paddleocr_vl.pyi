import abc
import torch
import torch.nn as nn
from .ernie45 import Ernie4_5ForCausalLM as Ernie4_5ForCausalLM
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
)
from .siglip import SiglipMLP as SiglipMLP
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    WeightsMapper as WeightsMapper,
    is_pp_missing_parameter as is_pp_missing_parameter,
    maybe_prefix as maybe_prefix,
)
from .vision import get_vit_attn_backend as get_vit_attn_backend
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import (
    BaseImageProcessor as BaseImageProcessor,
    BatchFeature as BatchFeature,
    PretrainedConfig as PretrainedConfig,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling as BaseModelOutputWithPooling,
)
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import parallel_state as parallel_state
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.linear import (
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
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems as ImageProcessorItems,
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
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)

def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = ...,
    max_pixels: int = ...,
): ...

class PaddleOCRVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_image_processor(self, **kwargs: object): ...
    def get_supported_mm_limits(self): ...
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: BaseImageProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...

class PaddleOCRVLDummyInputsBuilder(BaseDummyInputsBuilder[PaddleOCRVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class PaddleOCRVLMultiModalProcessor(
    BaseMultiModalProcessor[PaddleOCRVLProcessingInfo]
): ...

class Projector(nn.Module):
    text_config: Incomplete
    vision_config: Incomplete
    merge_kernel_size: Incomplete
    hidden_size: Incomplete
    pre_norm: Incomplete
    linear_1: Incomplete
    act: Incomplete
    linear_2: Incomplete
    def __init__(
        self,
        text_config: PretrainedConfig,
        vision_config: PretrainedConfig,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, image_features: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor: ...

class PaddleOCRImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

class SiglipVisionEmbeddings(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    patch_embedding: Incomplete
    num_patches: Incomplete
    num_positions: Incomplete
    cache_position_embedding: Incomplete
    cache_position_count: Incomplete
    position_embedding: Incomplete
    packing_position_embedding: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
        is_after_patchify: bool = False,
    ) -> torch.Tensor: ...
    def fetch_position_embedding_lfu_cache(
        self, embeddings: torch.Tensor, h: int, w: int, max_cache: int = 20
    ): ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]]
        | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor: ...

def all_gather_interleave(
    local_tensor: torch.Tensor, hidden_size: int, tp_size: int
): ...

class SiglipAttention(nn.Module):
    tp_size: Incomplete
    tp_rank: Incomplete
    hidden_size_per_attention_head: Incomplete
    num_attention_heads_per_partition: Incomplete
    qkv_proj: Incomplete
    out_proj: Incomplete
    attn: Incomplete
    apply_rotary_emb: Incomplete
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor: ...

class SigLIPRotaryEmbedding(nn.Module):
    dim: Incomplete
    theta: Incomplete
    def __init__(self, dim: int, theta: float = 10000.0) -> None: ...
    def rope_init(self) -> None: ...
    def forward(self, seqlen: int) -> torch.Tensor: ...

class SiglipEncoderLayer(nn.Module):
    embed_dim: Incomplete
    layer_norm1: Incomplete
    self_attn: Incomplete
    layer_norm2: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor: ...

class SiglipEncoder(nn.Module):
    config: Incomplete
    attn_backend: Incomplete
    layers: Incomplete
    rotary_pos_emb: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @staticmethod
    def flatten_list(image_grid_thw): ...
    def forward(
        self,
        inputs_embeds,
        cu_seqlens: torch.Tensor | None = None,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]]
        | None = None,
        height_position_ids: torch.Tensor | None = None,
        width_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class SiglipVisionTransformer(nn.Module):
    config: Incomplete
    embeddings: Incomplete
    encoder: Incomplete
    post_layernorm: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool | None = False,
        position_ids: torch.Tensor | None = None,
        height_position_ids: torch.Tensor | None = None,
        width_position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class SiglipVisionModel(nn.Module):
    vision_model: Incomplete
    quant_config: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def forward(
        self,
        pixel_values,
        interpolate_pos_encoding: bool = False,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]]
        | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPooling: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class PaddleOCRVLForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsMRoPE, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    visual: Incomplete
    mlp_AR: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ): ...
    def encode_image(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor: ...
    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
