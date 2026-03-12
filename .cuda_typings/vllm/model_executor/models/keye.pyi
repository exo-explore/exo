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
from .siglip import SiglipMLP as SiglipMLP
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    is_pp_missing_parameter as is_pp_missing_parameter,
    maybe_prefix as maybe_prefix,
)
from .vision import is_vit_use_data_parallel as is_vit_use_data_parallel
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import BaseImageProcessor as BaseImageProcessor, PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature as BatchFeature
from transformers.modeling_outputs import (
    BaseModelOutput as BaseModelOutput,
    BaseModelOutputWithPooling as BaseModelOutputWithPooling,
)
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
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
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
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
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete

def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
): ...

class KeyeImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

class KeyeImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    image_embeds: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

KeyeImageInputs: TypeAlias = KeyeImagePixelInputs | KeyeImageEmbeddingInputs

class KeyeVideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: Annotated[torch.Tensor, None]
    video_grid_thw: Annotated[torch.Tensor, None]

class KeyeVideoEmbeddingInputs(TensorSchema):
    type: Literal["video_embeds"]
    video_embeds: Annotated[torch.Tensor, None]
    video_grid_thw: Annotated[torch.Tensor, None]

KeyeVideoInputs: TypeAlias = KeyeVideoPixelInputs | KeyeVideoEmbeddingInputs

class KeyeVisionEmbeddings(nn.Module):
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
        self, embeddings, h, w, max_cache: int = 20
    ): ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]]
        | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor: ...

def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    apply_rotary_emb: ApplyRotaryEmb,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class KeyeSiglipAttention(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scale: Incomplete
    qkv_proj: Incomplete
    out_proj: Incomplete
    attn: Incomplete
    apply_rotary_emb: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
        cu_seqlens: list[torch.Tensor] | None = None,
        rope_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor: ...

class SigLIPRotaryEmbedding(nn.Module):
    dim: Incomplete
    theta: Incomplete
    def __init__(self, dim: int, theta: float = 10000.0) -> None: ...
    def rope_init(self) -> None: ...
    def forward(self, seqlen: int) -> torch.Tensor: ...

class KeyeSiglipEncoderLayer(nn.Module):
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
        attention_mask: torch.Tensor,
        output_attentions: bool | None = False,
        cu_seqlens: list[torch.Tensor] | None = None,
        rope_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.FloatTensor]: ...

class KeyeSiglipEncoder(nn.Module):
    config: Incomplete
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
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cu_seqlens: list[torch.Tensor] | None = None,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]]
        | None = None,
        height_position_ids: torch.Tensor | None = None,
        width_position_ids: torch.Tensor | None = None,
        use_rope: bool | None = False,
        window_size: bool | None = -1,
        vision_or_text: str = "vision",
    ) -> BaseModelOutput: ...

class KeyeSiglipVisionTransformer(nn.Module):
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
        pixel_values,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = False,
        attention_mask: torch.Tensor | None = None,
        sample_indices: torch.Tensor | None = None,
        image_indices: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        height_position_ids: torch.Tensor | None = None,
        width_position_ids: torch.Tensor | None = None,
        cu_seqlens: list[torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        vision_return_embed_list: bool | None = False,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]]
        | None = None,
        return_pooler_output: bool | None = True,
        use_rope: bool | None = False,
        window_size: bool | None = -1,
    ) -> BaseModelOutputWithPooling: ...

class KeyeSiglipVisionModel(nn.Module):
    config_class = PretrainedConfig
    main_input_name: str
    vision_model: Incomplete
    quant_config: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def forward(
        self,
        pixel_values,
        sample_indices: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
        position_ids: torch.Tensor | None = None,
        vision_return_embed_list: bool | None = False,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]]
        | None = None,
        cu_seqlens: list[torch.Tensor] | None = None,
        return_pooler_output: bool | None = True,
        use_rope: bool | None = False,
        window_size: bool | None = -1,
    ) -> BaseModelOutputWithPooling: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Projector(nn.Module):
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
        image_features: torch.Tensor | list[torch.Tensor],
        image_grid_thw: list[tuple[int, int, int]],
    ) -> torch.Tensor | list[torch.Tensor]: ...

class KeyeMultiModalDataParser(MultiModalDataParser): ...

class KeyeProcessingInfo(BaseProcessingInfo):
    def get_max_image_size(self) -> int: ...
    def get_max_frame_per_video(self) -> int: ...
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
    def get_num_frames_with_most_features(self, seq_len: int) -> int: ...
    def get_max_video_tokens(self, seq_len: int) -> int: ...

class KeyeBaseDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class KeyeDummyInputsBuilder(KeyeBaseDummyInputsBuilder[KeyeProcessingInfo]): ...
class KeyeMultiModalProcessor(BaseMultiModalProcessor[KeyeProcessingInfo]): ...

class BaseKeyeModule(nn.Module, SupportsMultiModal, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    visual: Incomplete
    mlp_AR: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None: ...
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

class KeyeForConditionalGeneration(
    BaseKeyeModule,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsMRoPE,
    metaclass=abc.ABCMeta,
):
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
