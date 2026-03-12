import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
)
from .utils import (
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from functools import cached_property as cached_property
from transformers import ChameleonConfig, ChameleonVQVAEConfig as ChameleonVQVAEConfig
from typing import Annotated, Any, Literal
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
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
    row_parallel_weight_loader as row_parallel_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems as MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete

class ChameleonImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, None]

class ChameleonProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(self) -> int: ...

class ChameleonDummyInputsBuilder(BaseDummyInputsBuilder[ChameleonProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class ChameleonMultiModalProcessor(
    BaseMultiModalProcessor[ChameleonProcessingInfo]
): ...

class ChameleonLayerNorm(nn.LayerNorm):
    normalized_shape: Incomplete
    def __init__(self, hidden_size, *args, **kwargs) -> None: ...
    def forward(self, hidden_states): ...

class ChameleonMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x): ...

class ChameleonAttention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    max_position_embeddings: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 4096,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class ChameleonDecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self,
        config: ChameleonConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class ChameleonSwinDecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self,
        config: ChameleonConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class ChameleonVQVAEVectorQuantizer(nn.Module):
    num_embeddings: Incomplete
    embedding_dim: Incomplete
    beta: Incomplete
    embedding: Incomplete
    re_embed: Incomplete
    def __init__(self, config: ChameleonVQVAEConfig) -> None: ...
    def forward(self, hidden_state: torch.Tensor): ...

class ChameleonVQVAEEncoderConvDownsample(nn.Module):
    conv: Incomplete
    def __init__(self, in_channels: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class ChameleonVQVAEEncoderResnetBlock(nn.Module):
    in_channels: Incomplete
    out_channels: Incomplete
    use_conv_shortcut: Incomplete
    norm1: Incomplete
    conv1: Incomplete
    norm2: Incomplete
    dropout: Incomplete
    conv2: Incomplete
    conv_shortcut: Incomplete
    nin_shortcut: Incomplete
    def __init__(
        self,
        config: ChameleonVQVAEConfig,
        in_channels: int,
        out_channels=None,
        conv_shortcut: bool = False,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class ChameleonVQVAEEncoderAttnBlock(nn.Module):
    in_channels: Incomplete
    norm: Incomplete
    q: Incomplete
    k: Incomplete
    v: Incomplete
    proj_out: Incomplete
    def __init__(self, in_channels: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class ChameleonVQVAEEncoder(nn.Module):
    num_resolutions: Incomplete
    num_res_blocks: Incomplete
    conv_in: Incomplete
    in_channel_multiplier: Incomplete
    down: Incomplete
    mid: Incomplete
    norm_out: Incomplete
    conv_out: Incomplete
    def __init__(self, config: ChameleonVQVAEConfig) -> None: ...
    def forward(self, pixel_values: torch.Tensor): ...

class ChameleonVQVAE(nn.Module):
    encoder: Incomplete
    quantize: Incomplete
    quant_conv: Incomplete
    post_quant_conv: Incomplete
    def __init__(self, config: ChameleonVQVAEConfig) -> None: ...
    def encode(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class ChameleonImageVocabularyMapping:
    vocab_map: Incomplete
    image_token_id: Incomplete
    def __init__(self, vocab_map: dict[str, int]) -> None: ...
    @cached_property
    def val2name(self): ...
    @cached_property
    def image_tokens(self): ...
    @cached_property
    def bpe2img(self): ...
    @cached_property
    def img2bpe(self): ...
    @cached_property
    def bpe2img_search_tensors(self): ...
    @cached_property
    def img2bpe_mapping_tensor(self): ...
    def convert_img2bpe(self, img_batch: torch.Tensor) -> torch.Tensor: ...

class ChameleonModel(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    vocabulary_mapping: Incomplete
    norm: Incomplete
    vqmodel: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def get_image_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...

class ChameleonForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
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
