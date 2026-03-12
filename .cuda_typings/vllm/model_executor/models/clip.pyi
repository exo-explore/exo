import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
)
from .interfaces_base import default_pooling_type as default_pooling_type
from .utils import AutoWeightsLoader as AutoWeightsLoader, maybe_prefix as maybe_prefix
from .vision import (
    VisionEncoderInfo as VisionEncoderInfo,
    VisionFeatureSelectStrategy as VisionFeatureSelectStrategy,
    VisionFeatureSelectStrategyStr as VisionFeatureSelectStrategyStr,
    get_num_selected_vision_tokens as get_num_selected_vision_tokens,
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    resolve_visual_encoder_outputs as resolve_visual_encoder_outputs,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Mapping
from functools import cached_property as cached_property
from transformers import (
    BatchFeature as BatchFeature,
    CLIPTextConfig as CLIPTextConfig,
    CLIPVisionConfig,
)
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import (
    Attention as Attention,
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler as DispatchPooler
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.interfaces import SupportsQuant as SupportsQuant
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalInputs as MultiModalInputs,
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
    ProcessorInputs as ProcessorInputs,
    PromptIndexTargets as PromptIndexTargets,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    TimingContext as TimingContext,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class CLIPImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, None]

class CLIPEncoderInfo(VisionEncoderInfo[CLIPVisionConfig]):
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...
    def get_image_size(self) -> int: ...
    def get_patch_size(self) -> int: ...
    def get_patch_grid_length(self) -> int: ...

class CLIPProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_vision_encoder_info(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_max_image_tokens(self) -> int: ...

class CLIPDummyInputsBuilder(BaseDummyInputsBuilder[CLIPProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class CLIPMultiModalProcessor(BaseMultiModalProcessor[CLIPProcessingInfo]):
    @cached_property
    def image_token_id(self) -> int: ...
    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInputs: ...

class CLIPTextEmbeddings(nn.Module):
    token_embedding: Incomplete
    position_embedding: Incomplete
    def __init__(self, config: CLIPTextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class CLIPVisionEmbeddings(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    class_embedding: Incomplete
    patch_embedding: Incomplete
    num_patches: Incomplete
    num_positions: Incomplete
    position_embedding: Incomplete
    def __init__(self, config: CLIPVisionConfig) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

class CLIPAttention(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    qkv_proj: Incomplete
    out_proj: Incomplete
    tp_size: Incomplete
    num_heads_per_partition: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
        attn_cls: type[Attention] | type[MMEncoderAttention],
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class CLIPMLP(nn.Module):
    config: Incomplete
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class CLIPEncoderLayer(nn.Module):
    self_attn: Incomplete
    layer_norm1: Incomplete
    mlp: Incomplete
    layer_norm2: Incomplete
    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
        attn_cls: type[Attention] | type[MMEncoderAttention],
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class CLIPEncoder(nn.Module):
    config: Incomplete
    layers: Incomplete
    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        *,
        prefix: str = "",
        attn_cls: type[Attention] | type[MMEncoderAttention],
    ) -> None: ...
    def forward(
        self, inputs_embeds: torch.Tensor, return_all_hidden_states: bool
    ) -> torch.Tensor | list[torch.Tensor]: ...

class CLIPTextTransformer(nn.Module):
    config: Incomplete
    embeddings: Incomplete
    encoder: Incomplete
    final_layer_norm: Incomplete
    def __init__(
        self,
        config: CLIPTextConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class CLIPVisionTransformer(nn.Module):
    config: Incomplete
    embeddings: Incomplete
    pre_layrnorm: Incomplete
    encoder: Incomplete
    post_layernorm: Incomplete
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self): ...
    @property
    def device(self): ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        select_layers: list[int] | None = None,
        feature_select_strategy: VisionFeatureSelectStrategy | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class CLIPVisionModel(nn.Module):
    vision_model: Incomplete
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        select_layers: list[int] | None = None,
        feature_select_strategy: VisionFeatureSelectStrategy | None = None,
    ) -> torch.Tensor: ...
    @property
    def dtype(self): ...
    @property
    def device(self): ...

class CLIPEmbeddingModel(
    nn.Module, SupportsMultiModal, SupportsQuant, metaclass=abc.ABCMeta
):
    is_pooling_model: bool
    packed_modules_mapping: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    projection_dim: Incomplete
    text_embed_dim: Incomplete
    vision_embed_dim: Incomplete
    text_model: Incomplete
    text_projection: Incomplete
    vision_model: Incomplete
    visual_projection: Incomplete
    pooler_config: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def get_text_features(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        feature_select_strategy: VisionFeatureSelectStrategy | None = None,
    ) -> torch.Tensor: ...
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
