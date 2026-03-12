import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsEagle3 as SupportsEagle3,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
    SupportsXDRoPE as SupportsXDRoPE,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import is_vit_use_data_parallel as is_vit_use_data_parallel
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Mapping
from transformers import BatchFeature as BatchFeature
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import parallel_state as parallel_state
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
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
from vllm.transformers_utils.configs.hunyuan_vl import (
    HunYuanVLConfig as HunYuanVLConfig,
    HunYuanVLVisionConfig as HunYuanVLVisionConfig,
)
from vllm.transformers_utils.processors.hunyuan_vl import (
    HunYuanVLProcessor as HunYuanVLProcessor,
)
from vllm.transformers_utils.processors.hunyuan_vl_image import (
    HunYuanVLImageProcessor as HunYuanVLImageProcessor,
    smart_resize as smart_resize,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete

class HunYuanVLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

class HunYuanVLImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    image_embeds: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

HunYuanVLImageInputs: TypeAlias = (
    HunYuanVLImagePixelInputs | HunYuanVLImageEmbeddingInputs
)

class HunYuanVisionMLP(nn.Module):
    dense_h_to_4h: Incomplete
    dense_4h_to_h: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = ...,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor): ...

class HunYuanVisionAttention(nn.Module):
    tp_size: Incomplete
    hidden_size_per_attention_head: Incomplete
    num_attention_heads_per_partition: Incomplete
    qkv: Incomplete
    o_proj: Incomplete
    scale: Incomplete
    attn: Incomplete
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class HunYuanVisionBlock(nn.Module):
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    self_attn: Incomplete
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
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class HunYuanVisionPatchEmbed(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    patch_size: Incomplete
    num_channels: Incomplete
    spatial_merge_size: Incomplete
    interpolate_mode: Incomplete
    patch_embedding: Incomplete
    max_num_patches: Incomplete
    num_positions: Incomplete
    position_edge: Incomplete
    position_embedding: Incomplete
    patch_pos_embed: Incomplete
    def __init__(self, config: HunYuanVLVisionConfig) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor: ...

class HunYuanVisionPatchMerger(nn.Module):
    spatial_merge_size: Incomplete
    proj: Incomplete
    mlp: Incomplete
    image_newline: Incomplete
    image_begin: Incomplete
    image_end: Incomplete
    image_sep: Incomplete
    before_rms: Incomplete
    after_rms: Incomplete
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_merge_size: int = 2,
        rms_norm_eps: float = 1e-05,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x, size=(16, 16)): ...

class HunYuanVisionTransformer(nn.Module):
    hidden_size: Incomplete
    num_heads: Incomplete
    spatial_merge_size: Incomplete
    embeddings: Incomplete
    layers: Incomplete
    perceive: Incomplete
    def __init__(
        self,
        vision_config: HunYuanVLVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def forward(self, x: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class HunYuanVLMultiModalDataParser(MultiModalDataParser): ...

class HunYuanVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> HunYuanVLProcessor: ...
    def get_image_processor(self, **kwargs: object) -> HunYuanVLImageProcessor: ...
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
        image_processor: HunYuanVLImageProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_max_image_tokens(self) -> int: ...

class HunYuanVLDummyInputsBuilder(BaseDummyInputsBuilder[HunYuanVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class HunYuanVLMultiModalProcessor(
    BaseMultiModalProcessor[HunYuanVLProcessingInfo]
): ...

class HunYuanVLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
    SupportsXDRoPE,
    SupportsEagle3,
    metaclass=abc.ABCMeta,
):
    hf_to_vllm_mapper: Incomplete
    supports_encoder_tp_data: bool
    def get_xdrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> torch.Tensor: ...
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    visual: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
