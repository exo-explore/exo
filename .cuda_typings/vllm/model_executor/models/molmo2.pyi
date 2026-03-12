import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    extract_layer_index as extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property as cached_property
from transformers import (
    BatchFeature,
    PretrainedConfig as PretrainedConfig,
    ProcessorMixin as ProcessorMixin,
    TensorType as TensorType,
)
from transformers.image_utils import ImageInput as ImageInput
from transformers.tokenization_utils_base import TextInput as TextInput
from transformers.video_utils import VideoInput as VideoInput
from typing import Annotated, Any
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.config.multimodal import (
    BaseDummyOptions as BaseDummyOptions,
    VideoDummyOptions as VideoDummyOptions,
)
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim as split_tensor_along_last_dim,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import (
    MulAndSilu as MulAndSilu,
    SiluAndMul as SiluAndMul,
    get_act_fn as get_act_fn,
)
from vllm.model_executor.layers.attention import (
    Attention as Attention,
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
    VideoItem as VideoItem,
)
from vllm.multimodal.parse import (
    ImageProcessorItems as ImageProcessorItems,
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.multimodal.processing.dummy_inputs import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.math_utils import round_down as round_down
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete
IMAGE_PROMPT: str
VIDEO_PROMPT: str

class Molmo2ImageInputs(TensorSchema):
    pixel_values: Annotated[torch.Tensor, None]
    token_pooling: Annotated[torch.Tensor, None]
    num_pooled_patches: Annotated[torch.Tensor, None]
    image_tokens: Annotated[torch.BoolTensor, None]
    num_image_tokens: Annotated[torch.Tensor, None]

class Molmo2VideoInputs(TensorSchema):
    pixel_values_videos: Annotated[torch.Tensor, None]
    token_pooling: Annotated[torch.Tensor, None]
    num_pooled_patches: Annotated[torch.Tensor, None]
    video_tokens: Annotated[torch.BoolTensor, None]
    num_video_tokens: Annotated[torch.Tensor, None]

@dataclass
class VitConfig:
    hidden_size: int = ...
    intermediate_size: int = ...
    num_hidden_layers: int = ...
    num_attention_heads: int = ...
    num_key_value_heads: int = ...
    head_dim: int = ...
    hidden_act: str = ...
    layer_norm_eps: float = ...
    image_default_input_size: tuple[int, int] = ...
    image_patch_size: int = ...
    image_num_pos: int = ...
    def __post_init__(self) -> None: ...
    @property
    def image_num_patch(self): ...

@dataclass
class AdapterConfig:
    vit_layers: tuple[int, int] = ...
    pooling_attention_mask: bool = ...
    hidden_size: int = ...
    num_attention_heads: int = ...
    num_key_value_heads: int = ...
    head_dim: int = ...
    hidden_act: str = ...
    intermediate_size: int = ...
    text_hidden_size: int = ...

@dataclass
class TextConfig:
    hidden_size: int = ...
    num_attention_heads: int = ...
    num_key_value_heads: int = ...
    head_dim: int = ...
    vocab_size: int = ...
    additional_vocab_size: int = ...
    qkv_bias: bool = ...
    num_hidden_layers: int = ...
    intermediate_size: int = ...
    hidden_act: str = ...
    max_position_embeddings: int = ...
    rope_theta: float = ...
    use_qk_norm: bool = ...
    qk_norm_type: str = ...
    layer_norm_eps: float = ...
    norm_after: bool = ...
    rope_scaling_layers: tuple[int, ...] | None = ...

class ViTMLP(nn.Module):
    w1: Incomplete
    act: Incomplete
    w2: Incomplete
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class ViTMultiHeadDotProductAttention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    merged_qkv: Incomplete
    wo: Incomplete
    scale: Incomplete
    attn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        use_bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: ...

class Molmo2VisionBlock(nn.Module):
    attention: Incomplete
    feed_forward: Incomplete
    attention_norm: Incomplete
    ffn_norm: Incomplete
    def __init__(
        self,
        config: VitConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Molmo2VisionBlockCollection(nn.Module):
    resblocks: Incomplete
    def __init__(
        self,
        config: VitConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: ...

class Molmo2VisionTransformer(nn.Module):
    num_prefix_tokens: int
    patch_num: Incomplete
    positional_embedding: Incomplete
    patch_embedding: Incomplete
    transformer: Incomplete
    def __init__(
        self,
        config: VitConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor: ...
    def forward(
        self, x: torch.Tensor, patch_num: int | None = None
    ) -> list[torch.Tensor]: ...

class ImagePoolingAttention(nn.Module):
    input_dim: Incomplete
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    kv_size: Incomplete
    q_proj: Incomplete
    merged_kv: Incomplete
    o_proj: Incomplete
    scale: Incomplete
    use_pytorch_sdpa: Incomplete
    attn: Incomplete
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        use_bias: bool = True,
        use_pytorch_sdpa: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class ImageProjectorMLP(nn.Module):
    merged_linear: Incomplete
    act_fn: Incomplete
    down_proj: Incomplete
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Molmo2VisionBackbone(nn.Module, SupportsQuant):
    packed_modules_mapping: Incomplete
    vit_config: Incomplete
    adapter_config: Incomplete
    vit_layers: Incomplete
    image_vit: Incomplete
    num_prefix_tokens: int
    image_pooling_2d: Incomplete
    image_projector: Incomplete
    def __init__(
        self,
        vit_config: VitConfig,
        adapter_config: AdapterConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def encode_image(self, images: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self, images: torch.Tensor, token_pooling: torch.Tensor
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Molmo2Attention(nn.Module):
    hidden_size: Incomplete
    tp_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    max_position_embeddings: Incomplete
    rope_theta: Incomplete
    qkv_proj: Incomplete
    tp_rank: int | None
    k_norm: nn.Module | None
    q_norm: nn.Module | None
    qk_norm_type: str | None
    rotary_emb: Incomplete
    scaling: Incomplete
    attn: Incomplete
    o_proj: Incomplete
    def __init__(
        self,
        config: TextConfig,
        rope_parameters: dict[str, Any],
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, **kwargs: object
    ) -> torch.Tensor: ...

class LanguageModelMLP(nn.Module):
    up_gate_proj: Incomplete
    act_fn: Incomplete
    down_proj: Incomplete
    def __init__(
        self,
        input_dim: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Molmo2DecoderLayer(nn.Module):
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self,
        config: TextConfig,
        rope_parameters: dict[str, Any],
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]: ...

class Molmo2DecoderNormAfterLayer(Molmo2DecoderLayer):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]: ...

class Molmo2TextModel(nn.Module, SupportsQuant):
    config: Incomplete
    embedding_size: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

def get_patches_grid_size(
    *, image_h: int, image_w: int, patch_size: int, pool_h: int, pool_w: int
) -> tuple[int, int]: ...
def get_candidate_tilings(max_num: int) -> list[tuple[int, int]]: ...
def select_tiling(
    *, height: int, width: int, patch_size: int, max_num_patches: int
): ...
def get_image_size(image: ImageInput) -> ImageSize: ...
def exif_transpose(images: ImageInput | None) -> ImageInput | None: ...
def build_flat_image_bool_length(
    image_grids: torch.LongTensor,
    image_patch_id: int,
    low_res_image_start_id: int,
    image_start_id: int,
    image_col_id: int,
    image_end_id: int,
) -> tuple[torch.LongTensor, torch.LongTensor]: ...
def build_flat_video_bool_length(
    video_grids: torch.LongTensor,
    image_patch_id: int,
    frame_start_id: int,
    frame_end_id: int,
) -> tuple[torch.LongTensor, torch.LongTensor]: ...

class Molmo2ProcessorWrapper:
    processor: Incomplete
    hf_config: Incomplete
    def __init__(
        self, processor: ProcessorMixin, hf_config: PretrainedConfig
    ) -> None: ...
    @cached_property
    def vocab(self) -> dict[str, int]: ...
    @cached_property
    def max_crops(self) -> int: ...
    @cached_property
    def image_pooling_h(self) -> int: ...
    @cached_property
    def image_pooling_w(self) -> int: ...
    @cached_property
    def video_pooling_h(self) -> int: ...
    @cached_property
    def video_pooling_w(self) -> int: ...
    @cached_property
    def base_image_input_size(self) -> tuple[int, int]: ...
    @cached_property
    def image_patch_size(self) -> int: ...
    @cached_property
    def overlap_margins(self) -> tuple[int, int]: ...
    @cached_property
    def bos_token(self) -> str: ...
    @cached_property
    def image_patch_id(self) -> int: ...
    @cached_property
    def im_col_id(self) -> int: ...
    @cached_property
    def im_start_id(self) -> int: ...
    @cached_property
    def im_end_id(self) -> int: ...
    @cached_property
    def low_res_im_start_id(self) -> int: ...
    @cached_property
    def frame_start_id(self) -> int: ...
    @cached_property
    def frame_end_id(self) -> int: ...
    @cached_property
    def im_low_res_id(self) -> int: ...
    @cached_property
    def image_placeholder_id(self) -> int: ...
    @cached_property
    def video_placeholder_id(self) -> int: ...
    @cached_property
    def image_token_ids(self) -> list[int]: ...
    def select_tiling(
        self, *, image_height: int, image_width: int
    ) -> tuple[int, int]: ...
    def get_base_grid_size(self, is_video: bool) -> tuple[int, int]: ...
    def get_patches_grid_size(
        self, *, image_height: int, image_width: int
    ) -> tuple[int, int]: ...
    def __call__(
        self,
        text: TextInput | list[TextInput] | None = None,
        images: ImageInput | None = None,
        videos: VideoInput | None = None,
        return_tensors: str | TensorType = None,
        **kwargs: object,
    ) -> BatchFeature: ...

def get_candidate_target_fps(
    video_fps: int | float, sampling_fps: int | float, max_fps: int | float = ...
) -> list[float]: ...
def get_target_fps(
    video_fps: float,
    max_frames: int,
    total_frames: int,
    frame_sample_mode: str,
    candidate_target_fps: list[float],
) -> float | None: ...
def get_frame_times_and_chosen_fps(
    selected_target_fps, total_frames, max_frames, video_fps
): ...

class Molmo2ProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self): ...
    def get_hf_processor(self, **kwargs: object) -> Molmo2ProcessorWrapper: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(
        self, *, image_height: int, image_width: int, processor: Molmo2ProcessorWrapper
    ) -> int: ...
    def get_num_video_tokens(
        self, *, num_frames: int, processor: Molmo2ProcessorWrapper
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...

class Molmo2DummyInputsBuilder(BaseDummyInputsBuilder[Molmo2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Molmo2MultiModalProcessor(BaseMultiModalProcessor[Molmo2ProcessingInfo]): ...

class Molmo2ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
    SupportsQuant,
    metaclass=abc.ABCMeta,
):
    hf_to_vllm_mapper: Incomplete
    packed_modules_mapping: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    vision_backbone: Incomplete
    model: Incomplete
    img_patch_id: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @property
    def dtype(self): ...
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
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
