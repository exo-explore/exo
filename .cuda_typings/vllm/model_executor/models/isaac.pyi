import PIL.Image
import abc
import torch
import torch.nn as nn
from .vision import is_vit_use_data_parallel as is_vit_use_data_parallel
from _typeshed import Incomplete
from collections.abc import Iterable, Iterator, Mapping
from transformers.image_processing_utils import BatchFeature
from transformers.utils import TensorType as TensorType
from typing import Annotated, Any
from typing_extensions import TypedDict, Unpack
from vllm.config import VllmConfig as VllmConfig
from vllm.config.model import ModelConfig as ModelConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import parallel_state as parallel_state
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
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
from vllm.model_executor.models.siglip import SiglipMLP as SiglipMLP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import get_tokenizer as get_tokenizer
from vllm.tokenizers.hf import get_cached_tokenizer as get_cached_tokenizer
from vllm.transformers_utils.config import (
    patch_rope_parameters as patch_rope_parameters,
)
from vllm.transformers_utils.configs import (
    IsaacConfig as IsaacConfig,
    PixelShuffleSiglip2VisionConfig as PixelShuffleSiglip2VisionConfig,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

def create_cumulative_seq_lengths(
    seq_sizes: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]: ...

class Siglip2VariableSequenceEmbeddings(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    patch_size: Incomplete
    patch_embedding: Incomplete
    num_patches: Incomplete
    position_embedding_size: Incomplete
    position_embedding: Incomplete
    def __init__(self, config: PixelShuffleSiglip2VisionConfig) -> None: ...
    def positional_embeddings(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor: ...
    def forward(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ): ...

def create_pixel_shuffle_index_map(
    seq_sizes: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor: ...
def pixel_shuffle_varlen(
    x: torch.Tensor, token_grids: torch.Tensor, scale_factor: int = 1
) -> torch.Tensor: ...

MAX_PIXELS: int
VISION_MEAN: Incomplete
VISION_STD: Incomplete
VISION_SCALE: Incomplete

def extract_image_pil(image: PIL.Image.Image) -> torch.Tensor | None: ...
def get_image_size_for_max_num_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: int | None = None,
    eps: float = 1e-05,
    pixel_shuffle_scale: int = 1,
) -> tuple[int, int]: ...
def prepare_image_tensor(image: torch.Tensor, scale: float = ...) -> torch.Tensor: ...
def patchify_vision(image: torch.Tensor, patch_size: int) -> torch.Tensor: ...
def process_vision_for_patches(
    images: torch.Tensor,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: int | None = None,
    pixel_shuffle_scale: int = 1,
) -> tuple[torch.Tensor, list[int]]: ...

class IsaacImageProcessorKwargs(TypedDict, total=False):
    patch_size: int
    max_num_patches: int
    min_num_patches: int
    pixel_shuffle_scale: int

class IsaacImageProcessor:
    patch_size: int
    max_num_patches: int
    min_num_patches: int
    pixel_shuffle_scale: int
    valid_kwargs = IsaacImageProcessorKwargs
    model_input_names: Incomplete
    vision_max_num_patches: Incomplete
    vision_min_num_patches: Incomplete
    def __init__(self, kwargs) -> None: ...
    def preprocess(
        self,
        images: list[torch.Tensor],
        return_tensors: str | TensorType | None,
        **kwargs: Unpack[IsaacImageProcessorKwargs],
    ) -> BatchFeature: ...

class IsaacProcessor:
    image_token: Incomplete
    image_processor: Incomplete
    tokenizer: Incomplete
    def __init__(self, image_processor=None, tokenizer=None, **kwargs) -> None: ...
    def __call__(self, text=None, images=None, **kwargs) -> BatchFeature: ...
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Any: ...

class IsaacProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> IsaacConfig: ...
    def get_hf_processor(self, **kwargs) -> IsaacProcessor: ...
    def get_tokenizer(self): ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_image_processor(self, **kwargs) -> IsaacImageProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]: ...

class IsaacDummyInputsBuilder(BaseDummyInputsBuilder[IsaacProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class IsaacImagePixelInputs(TensorSchema):
    pixel_values: Annotated[torch.Tensor, None]
    image_grid_thw: Annotated[torch.Tensor, None]

class IsaacMultiModalProcessor(BaseMultiModalProcessor): ...

class Siglip2VisionAttention(nn.Module):
    tp_size: Incomplete
    tp_rank: Incomplete
    hidden_size_per_attention_head: Incomplete
    num_attention_heads_per_partition: Incomplete
    qkv_proj: Incomplete
    out_proj: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: PixelShuffleSiglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None: ...
    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor: ...

class Siglip2EncoderLayer(nn.Module):
    embed_dim: Incomplete
    layer_norm1: Incomplete
    self_attn: Incomplete
    layer_norm2: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        config: PixelShuffleSiglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor: ...

class Siglip2Encoder(nn.Module):
    config: Incomplete
    layers: Incomplete
    def __init__(
        self,
        config: PixelShuffleSiglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class Siglip2VisionTransformer(nn.Module):
    config: Incomplete
    quant_config: Incomplete
    embeddings: Incomplete
    pixel_shuffle_scale_factor: Incomplete
    encoder: Incomplete
    post_layernorm: Incomplete
    def __init__(
        self,
        config: PixelShuffleSiglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class IsaacVisionEmbedding(nn.Module):
    transformer: Incomplete
    linear_fc1: Incomplete
    act: Incomplete
    linear_fc2: Incomplete
    def __init__(
        self,
        vision_cfg: PixelShuffleSiglip2VisionConfig,
        hidden_dim: int,
        output_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor: ...

class IsaacForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsMRoPE,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    supports_encoder_tp_data: bool
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    vision_token_id: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    vision_embedding: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None: ...
    def iter_mm_grid_hw(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, int, int]]: ...
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
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
