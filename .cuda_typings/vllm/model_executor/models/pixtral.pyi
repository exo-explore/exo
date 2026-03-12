import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .module_mapping import MultiModelKeys as MultiModelKeys
from .utils import (
    StageMissingLayer as StageMissingLayer,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import (
    VisionEncoderInfo as VisionEncoderInfo,
    VisionFeatureSelectStrategy as VisionFeatureSelectStrategy,
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    resolve_visual_encoder_outputs as resolve_visual_encoder_outputs,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from transformers import PixtralVisionConfig
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import (
    get_act_and_mul_fn as get_act_and_mul_fn,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.multimodal import (
    MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    NestedTensors as NestedTensors,
)
from vllm.multimodal.parse import (
    ImageProcessorItems as ImageProcessorItems,
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder as BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    MultiModalProcessingInfo as MultiModalProcessingInfo,
    ProcessorInputs as ProcessorInputs,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
    TimingContext as TimingContext,
)
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config as cached_tokenizer_from_config
from vllm.tokenizers.mistral import MistralTokenizer as MistralTokenizer
from vllm.transformers_utils.processors.pixtral import (
    MistralCommonPixtralProcessor as MistralCommonPixtralProcessor,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

USE_XFORMERS_OPS: bool
PATCH_MERGE: str

class PixtralImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    images: Annotated[torch.Tensor | list[torch.Tensor], None]

class PixtralProcessingInfo(BaseProcessingInfo):
    def get_tokenizer(self) -> MistralTokenizer: ...
    def get_hf_processor(self, **kwargs) -> MistralCommonPixtralProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...

class PixtralDummyInputsBuilder(BaseDummyInputsBuilder[PixtralProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> ProcessorInputs: ...

class PixtralMultiModalProcessor(BaseMultiModalProcessor[PixtralProcessingInfo]): ...

class PixtralForConditionalGeneration(
    nn.Module, SupportsLoRA, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    vision_args: Incomplete
    language_model: Incomplete
    vision_encoder: Incomplete
    pre_mm_projector_norm: Incomplete
    patch_merger: Incomplete
    vision_language_adapter: Incomplete
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
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int: ...
    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int: ...

@dataclass
class VisionEncoderArgs:
    hidden_size: int
    num_channels: int
    image_size: int
    patch_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rope_theta: float
    image_token_id: int
    adapter_bias: bool = ...
    spatial_merge_size: int = ...
    add_pre_mm_projector_layer_norm: bool = ...
    mm_projector_id: str = ...

def precompute_freqs_cis_2d(
    dim: int, height: int, width: int, theta: float
) -> torch.Tensor: ...
def apply_rotary_emb_vit(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...

class FeedForward(nn.Module):
    w1: Incomplete
    w2: Incomplete
    w3: Incomplete
    def __init__(self, args: VisionEncoderArgs) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Attention(nn.Module):
    args: Incomplete
    n_heads: Incomplete
    head_dim: Incomplete
    wq: Incomplete
    wk: Incomplete
    wv: Incomplete
    wo: Incomplete
    def __init__(self, args: VisionEncoderArgs) -> None: ...
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor: ...

class TransformerBlock(nn.Module):
    attention: Incomplete
    feed_forward: Incomplete
    attention_norm: Incomplete
    ffn_norm: Incomplete
    def __init__(self, args: VisionEncoderArgs) -> None: ...
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor: ...

class Transformer(nn.Module):
    layers: Incomplete
    def __init__(self, args: VisionEncoderArgs) -> None: ...
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, freqs_cis: torch.Tensor | None
    ) -> torch.Tensor: ...

def position_meshgrid(patch_embeds_list: list[torch.Tensor]) -> torch.Tensor: ...

class VisionTransformer(nn.Module):
    args: Incomplete
    patch_conv: Incomplete
    ln_pre: Incomplete
    transformer: Incomplete
    def __init__(self, args: VisionEncoderArgs) -> None: ...
    @property
    def max_patches_per_side(self) -> int: ...
    @property
    def device(self) -> torch.types.Device: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def freqs_cis(self) -> torch.Tensor: ...
    def forward(self, images: list[torch.Tensor]) -> torch.Tensor: ...

class VisionLanguageAdapter(nn.Module):
    w_in: Incomplete
    gelu: Incomplete
    w_out: Incomplete
    def __init__(self, args: VisionEncoderArgs, dim: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class PatchMerger(nn.Module):
    spatial_merge_size: Incomplete
    mlp_input_dim: Incomplete
    merging_layer: Incomplete
    def __init__(
        self,
        vision_encoder_dim: int,
        spatial_merge_size: int,
        use_mlp_bias: bool = False,
    ) -> None: ...
    def forward(
        self, x: torch.Tensor, image_sizes: list[tuple[int, int]]
    ) -> torch.Tensor: ...
    def permute(
        self, x: torch.Tensor, image_sizes: list[tuple[int, int]]
    ) -> torch.Tensor: ...

def get_sub_grids(
    x: torch.Tensor, image_sizes: list[tuple[int, int]], spatial_merge_size: int
) -> list[torch.Tensor]: ...

class PixtralHFEncoderInfo(VisionEncoderInfo[PixtralVisionConfig]):
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...
    def get_image_size(self) -> int: ...
    def get_patch_size(self) -> int: ...
    def get_patch_grid_length(self) -> int: ...
    def get_patch_grid_size(
        self, *, image_width: int, image_height: int
    ) -> tuple[int, int]: ...

class PixtralHFMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_and_mul: Incomplete
    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class PixtralHFAttention(nn.Module):
    config: Incomplete
    total_num_heads: Incomplete
    head_dim: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    tp_size: Incomplete
    n_heads: Incomplete
    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class PixtralHFTransformerBlock(nn.Module):
    attention_norm: Incomplete
    attention: Incomplete
    feed_forward: Incomplete
    ffn_norm: Incomplete
    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor: ...

class PixtralHFTransformer(nn.Module):
    layers: Incomplete
    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        return_all_hidden_states: bool,
    ) -> torch.Tensor: ...

class PixtralHFVisionModel(nn.Module):
    config: Incomplete
    patch_conv: Incomplete
    ln_pre: Incomplete
    transformer: Incomplete
    dtype: Incomplete
    device: Incomplete
    patch_positional_embedding: Incomplete
    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        pixel_values: list[torch.Tensor],
        *,
        select_layers: list[int] | None = None,
        feature_select_strategy: VisionFeatureSelectStrategy | None = None,
    ) -> tuple[torch.Tensor, ...]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
