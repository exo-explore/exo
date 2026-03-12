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
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from transformers import (
    BaseImageProcessor as BaseImageProcessor,
    BatchFeature as BatchFeature,
    PretrainedConfig as PretrainedConfig,
)
from typing import Annotated
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim as split_tensor_along_last_dim,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import (
    MulAndSilu as MulAndSilu,
    QuickGELU as QuickGELU,
    SiluAndMul as SiluAndMul,
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
    PromptIndexTargets as PromptIndexTargets,
    PromptInsertion as PromptInsertion,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

VIT_LAYERS: Incomplete
NUM_PREFIX_TOKENS: int
ADDITIONAL_VOCAB_SIZE: int
IMAGE_PATCH_TOKEN: str
IM_COL_TOKEN: str
IM_START_TOKEN: str
IM_END_TOKEN: str
POOLING_SIZE: int

class MolmoImageInputs(TensorSchema):
    images: Annotated[torch.Tensor, None]
    image_masks: Annotated[torch.Tensor | None, None]
    image_input_idx: Annotated[torch.Tensor, None]
    num_crops: Annotated[torch.Tensor, None]

@dataclass
class VisionBackboneConfig:
    image_default_input_size: tuple[int, int] = ...
    image_patch_size: int = ...
    image_pos_patch_size: int = ...
    image_emb_dim: int = ...
    image_num_heads: int = ...
    image_num_key_value_heads: int = ...
    image_num_layers: int = ...
    image_mlp_dim: int = ...
    image_mlp_activations: str = ...
    image_num_pos: int = ...
    image_norm_eps: float = ...
    def __post_init__(self) -> None: ...
    @property
    def image_num_patch(self): ...

class ViTMLP(nn.Module):
    w1: Incomplete
    act: Incomplete
    w2: Incomplete
    def __init__(
        self,
        config: VisionBackboneConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MultiHeadDotProductAttention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    wq: Incomplete
    wk: Incomplete
    wv: Incomplete
    wo: Incomplete
    scale: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: VisionBackboneConfig,
        use_bias: bool = True,
        nlayers: int = 1,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, inputs_q: torch.Tensor, inputs_kv: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class ResidualAttentionBlock(nn.Module):
    attention: Incomplete
    feed_forward: Incomplete
    attention_norm: Incomplete
    ffn_norm: Incomplete
    def __init__(
        self,
        config: VisionBackboneConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class BlockCollection(nn.Module):
    resblocks: Incomplete
    def __init__(
        self,
        config: VisionBackboneConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: ...

class VisionTransformer(nn.Module):
    patch_num: Incomplete
    class_embedding: Incomplete
    num_prefix_tokens: int
    positional_embedding: Incomplete
    patch_embedding: Incomplete
    pre_ln: Incomplete
    transformer: Incomplete
    def __init__(
        self,
        config: VisionBackboneConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor: ...
    def forward(
        self, x: torch.Tensor, patch_num: int | None = None
    ) -> list[torch.Tensor]: ...

class MolmoAttention(nn.Module):
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
    qkv_proj: Incomplete
    tp_rank: int | None
    k_norm: nn.Module | None
    q_norm: nn.Module | None
    rotary_emb: Incomplete
    scaling: Incomplete
    attn: Incomplete
    o_proj: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class LanguageModelMLP(nn.Module):
    hidden_size: Incomplete
    intermediate_size: Incomplete
    gate_up_proj: Incomplete
    act_fn: Incomplete
    down_proj: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        input_dim: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class ImageProjectorMLP(nn.Module):
    hidden_size: Incomplete
    intermediate_size: Incomplete
    merged_linear: Incomplete
    act_fn: Incomplete
    down_proj: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        input_dim: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MolmoDecoderLayer(nn.Module):
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]: ...

class MolmoDecoderNormAfterLayer(MolmoDecoderLayer):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]: ...

class MolmoVisionBackbone(nn.Module, SupportsQuant):
    packed_modules_mapping: Incomplete
    vit_layers: Incomplete
    image_num_patch: Incomplete
    llm_patches_per_crop: Incomplete
    image_vit: Incomplete
    num_prefix_tokens: Incomplete
    image_pooling_2d: Incomplete
    image_projector: Incomplete
    pad_embed: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        vision_config: VisionBackboneConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def encode_image(self, images: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self, images: torch.Tensor, image_masks: torch.Tensor
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class MolmoModel(nn.Module, SupportsQuant):
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
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

def get_num_patches(
    num_tiles: int,
    *,
    crop_patches: int,
    left_margin: int,
    right_margin: int,
    pooling_size: int,
) -> int: ...
def get_patches_grid_size(
    *,
    tiling_h: int,
    tiling_w: int,
    crop_patches: int,
    left_margin: int,
    right_margin: int,
    pooling_size: int,
) -> tuple[int, int]: ...
def get_candidate_tilings(max_num: int) -> list[tuple[int, int]]: ...
def select_tiling(
    *, height: int, width: int, patch_size: int, max_num_patches: int
): ...

class MolmoProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def select_tiling(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: BaseImageProcessor,
    ) -> tuple[int, int]: ...
    def get_patches_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: BaseImageProcessor,
    ) -> tuple[int, int]: ...
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: BaseImageProcessor,
    ) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...

class MolmoDummyInputsBuilder(BaseDummyInputsBuilder[MolmoProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class MolmoMultiModalProcessor(BaseMultiModalProcessor[MolmoProcessingInfo]): ...

class MolmoForCausalLM(
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
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
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
