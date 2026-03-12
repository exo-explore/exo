import abc
import torch
import torch.nn as nn
from .blip import (
    BlipVisionModel as BlipVisionModel,
    get_blip_num_patches as get_blip_num_patches,
)
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
)
from .module_mapping import MultiModelKeys as MultiModelKeys
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import Blip2QFormerConfig as Blip2QFormerConfig
from typing import Annotated, Literal, TypeAlias
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
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
    PromptIndexTargets as PromptIndexTargets,
    PromptInsertion as PromptInsertion,
    PromptUpdate as PromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class Blip2ImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, None]

class Blip2ImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, None]

Blip2ImageInputs: TypeAlias = Blip2ImagePixelInputs | Blip2ImageEmbeddingInputs

class Blip2QFormerMultiHeadAttention(nn.Module):
    config: Incomplete
    num_attention_heads: Incomplete
    attention_head_size: Incomplete
    all_head_size: Incomplete
    scaling: Incomplete
    query: Incomplete
    key: Incomplete
    value: Incomplete
    position_embedding_type: Incomplete
    dropout: Incomplete
    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        is_cross_attention: bool = False,
        prefix: str = "",
    ) -> None: ...
    def transpose_for_scores(self, x): ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
    ): ...

class Blip2QFormerSelfOutput(nn.Module):
    dense: Incomplete
    LayerNorm: Incomplete
    dropout: Incomplete
    def __init__(self, config: Blip2QFormerConfig, prefix: str = "") -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class Blip2QFormerAttention(nn.Module):
    attention: Incomplete
    output: Incomplete
    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        is_cross_attention: bool = False,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
    ) -> tuple[torch.Tensor]: ...

class Blip2QFormerIntermediate(nn.Module):
    dense: Incomplete
    intermediate_act_fn: Incomplete
    def __init__(self, config: Blip2QFormerConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Blip2QFormerOutput(nn.Module):
    dense: Incomplete
    LayerNorm: Incomplete
    dropout: Incomplete
    def __init__(self, config: Blip2QFormerConfig, prefix: str = "") -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class Blip2QFormerLayer(nn.Module):
    chunk_size_feed_forward: Incomplete
    seq_len_dim: int
    attention: Incomplete
    layer_idx: Incomplete
    crossattention: Incomplete
    has_cross_attention: bool
    intermediate_query: Incomplete
    output_query: Incomplete
    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        layer_idx: int,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        query_length: int,
    ): ...
    def feed_forward_chunk(self, attention_output: torch.Tensor) -> torch.Tensor: ...
    def feed_forward_chunk_query(
        self, attention_output: torch.Tensor
    ) -> torch.Tensor: ...

class Blip2QFormerEncoder(nn.Module):
    config: Incomplete
    layer: Incomplete
    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        query_length: int,
    ) -> torch.Tensor: ...

class Blip2QFormerModel(nn.Module):
    config: Incomplete
    layernorm: Incomplete
    dropout: Incomplete
    encoder: Incomplete
    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, query_embeds: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor
    ) -> torch.Tensor: ...

class Blip2ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(self) -> int: ...

class Blip2DummyInputsBuilder(BaseDummyInputsBuilder[Blip2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Blip2MultiModalProcessor(BaseMultiModalProcessor[Blip2ProcessingInfo]): ...

class Blip2ForConditionalGeneration(
    nn.Module,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
    metaclass=abc.ABCMeta,
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    vision_model: Incomplete
    query_tokens: Incomplete
    qformer: Incomplete
    language_projection: Incomplete
    language_model: Incomplete
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
    ) -> IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int: ...
    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int: ...
