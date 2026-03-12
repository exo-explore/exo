import abc
import torch
import torch.nn as nn
from PIL import Image
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import (
    BartConfig as BartConfig,
    BatchFeature,
    PretrainedConfig as PretrainedConfig,
    TensorType as TensorType,
)
from typing import Annotated, Literal
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
)
from vllm.model_executor.models.radio import RadioModel as RadioModel
from vllm.model_executor.models.whisper import (
    WhisperAttention as WhisperAttention,
    WhisperCrossAttention as WhisperCrossAttention,
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
    BaseProcessingInfo as BaseProcessingInfo,
    EncDecMultiModalProcessor as EncDecMultiModalProcessor,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.transformers_utils.configs.radio import RadioConfig as RadioConfig
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)
from vllm.v1.attention.backend import AttentionType as AttentionType

logger: Incomplete
DEFAULT_FINAL_IMAGE_SIZE: Incomplete

class BartScaledWordEmbedding(VocabParallelEmbedding):
    embed_scale: Incomplete
    def __init__(
        self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...

class BartParallelLMHead(ParallelLMHead):
    embed_scale: Incomplete
    def __init__(
        self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...

class BartDecoderLayer(nn.Module):
    embed_dim: Incomplete
    self_attn: Incomplete
    activation_fn: Incomplete
    self_attn_layer_norm: Incomplete
    encoder_attn: Incomplete
    encoder_attn_layer_norm: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    final_layer_norm: Incomplete
    def __init__(
        self,
        config: BartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class MBartDecoderLayer(BartDecoderLayer):
    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class MBartDecoderNoPos(nn.Module):
    cache_config: Incomplete
    quant_config: Incomplete
    lora_config: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    layernorm_embedding: Incomplete
    layer_norm: Incomplete
    def __init__(
        self,
        config: BartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        lora_config: LoRAConfig | None = None,
        embed_tokens: nn.Embedding | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        decoder_input_ids: torch.Tensor | None,
        *,
        encoder_hidden_states: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class NemotronParsePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, None]

class NemotronParseImageProcessor:
    final_size: Incomplete
    norm_mean: Incomplete
    norm_std: Incomplete
    def __init__(self, final_size: tuple = ..., **kwargs) -> None: ...
    def preprocess(
        self, images: Image.Image | list[Image.Image], **kwargs
    ) -> dict[str, torch.Tensor]: ...
    def __call__(
        self, images: Image.Image | list[Image.Image], **kwargs
    ) -> dict[str, torch.Tensor]: ...

class NemotronParseProcessor:
    config: Incomplete
    tokenizer: Incomplete
    image_processor: Incomplete
    def __init__(
        self, config: PretrainedConfig, tokenizer: TokenizerLike, **kwargs
    ) -> None: ...
    def __call__(
        self,
        text: str | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature: ...

class NemotronParseProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs) -> NemotronParseProcessor: ...
    def get_default_tok_params(self) -> TokenizeParams: ...
    @property
    def skip_prompt_length_check(self) -> bool: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(self) -> int: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None: ...

class NemotronParseDummyInputsBuilder(
    BaseDummyInputsBuilder[NemotronParseProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class NemotronParseMultiModalProcessor(
    EncDecMultiModalProcessor[NemotronParseProcessingInfo]
):
    def create_encoder_prompt(
        self, prompt: str | list[int], mm_items: MultiModalDataItems
    ) -> str | list[int]: ...

class RadioWithNeck(nn.Module):
    config: Incomplete
    model_encoder: Incomplete
    conv1: Incomplete
    layer_norm1: Incomplete
    conv2: Incomplete
    layer_norm2: Incomplete
    sum_proj: Incomplete
    layer_norm3: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def get_vit_model_from_radio_config(
        self,
        hf_config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> RadioModel: ...
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...

class NemotronParseForConditionalGeneration(
    nn.Module, SupportsMultiModal, metaclass=abc.ABCMeta
):
    config: Incomplete
    vision_config: Incomplete
    encoder: Incomplete
    decoder: Incomplete
    vocab_size: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
