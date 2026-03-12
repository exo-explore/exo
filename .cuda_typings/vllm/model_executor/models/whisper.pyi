import abc
import enum
import numpy as np
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsTranscription as SupportsTranscription,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    cast_overflow_tensors as cast_overflow_tensors,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from torch import nn
from transformers import (
    BatchFeature as BatchFeature,
    WhisperConfig,
    WhisperFeatureExtractor,
)
from typing import Annotated, Literal
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.inputs.data import (
    ExplicitEncoderDecoderPrompt as ExplicitEncoderDecoderPrompt,
    PromptType as PromptType,
    TextPrompt as TextPrompt,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import (
    Attention as Attention,
    CrossAttention as CrossAttention,
    MMEncoderAttention as MMEncoderAttention,
)
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
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.whisper_utils import (
    ISO639_1_SUPPORTED_LANGS as ISO639_1_SUPPORTED_LANGS,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseProcessingInfo as BaseProcessingInfo,
    EncDecMultiModalProcessor as EncDecMultiModalProcessor,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.transformers_utils.processor import (
    cached_processor_from_config as cached_processor_from_config,
)
from vllm.utils.jsontree import json_map_leaves as json_map_leaves
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)
from vllm.utils.torch_utils import set_default_torch_dtype as set_default_torch_dtype
from vllm.v1.attention.backend import AttentionType as AttentionType

logger: Incomplete

class WhisperPosEmbedType(enum.Enum):
    SINUSOIDAL = "sinusoidal"
    ROPE = "rope"
    LEARNED = "learned"

class WhisperAudioInputs(TensorSchema):
    input_features: Annotated[list[torch.Tensor] | None, None]

class WhisperEncoderAttention(MMEncoderAttention):
    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor: ...

class WhisperPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int) -> None: ...
    def forward(self, position_ids): ...

class WhisperAttention(nn.Module):
    embed_dim: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    attn_type: Incomplete
    scaling: Incomplete
    out_proj: Incomplete
    attn: Incomplete
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_type: AttentionType = ...,
        per_layer_sliding_window: int | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class WhisperCrossAttention(WhisperAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None
    ): ...

class WhisperMLP(nn.Module):
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        act_fn: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class WhisperEncoderLayer(nn.Module):
    embed_dim: Incomplete
    self_attn: Incomplete
    self_attn_layer_norm: Incomplete
    mlp: Incomplete
    final_layer_norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class WhisperDecoderLayer(nn.Module):
    self_attn: Incomplete
    self_attn_layer_norm: Incomplete
    encoder_attn: Incomplete
    encoder_attn_layer_norm: Incomplete
    mlp: Incomplete
    final_layer_norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None
    ): ...

class WhisperEncoder(nn.Module):
    pos_embed_type: Incomplete
    num_mel_bins: Incomplete
    max_source_positions: Incomplete
    embed_scale: Incomplete
    conv1: Incomplete
    conv2: Incomplete
    total_stride: Incomplete
    layer_norm: Incomplete
    embed_positions: Incomplete
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", init_in_fp32: bool = False
    ) -> None: ...
    def forward(
        self, input_features: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor: ...

class WhisperDecoder(nn.Module):
    layerdrop: Incomplete
    padding_idx: Incomplete
    max_target_positions: Incomplete
    max_source_positions: Incomplete
    embed_scale: Incomplete
    embed_tokens: Incomplete
    embed_positions: Incomplete
    layer_norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ): ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...

class WhisperModel(nn.Module):
    encoder: Incomplete
    decoder: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor],
    ) -> torch.Tensor: ...
    def get_encoder_outputs(
        self, input_features: torch.Tensor | list[torch.Tensor] | None
    ) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class WhisperProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> WhisperConfig: ...
    def get_default_tok_params(self) -> TokenizeParams: ...
    def get_data_parser(self): ...
    @property
    def skip_prompt_length_check(self) -> bool: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor: ...
    def get_target_channels(self) -> int: ...
    def get_num_audio_tokens(self) -> int: ...

class WhisperDummyInputsBuilder(BaseDummyInputsBuilder[WhisperProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class WhisperMultiModalProcessor(EncDecMultiModalProcessor[WhisperProcessingInfo]):
    def create_encoder_prompt(
        self, prompt: str | list[int], mm_items: MultiModalDataItems
    ) -> str | list[int]: ...

class WhisperForConditionalGeneration(
    nn.Module,
    SupportsTranscription,
    SupportsMultiModal,
    SupportsLoRA,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    supports_transcription_only: bool
    supports_segment_timestamp: bool
    supports_explicit_language_detection: bool
    supported_languages = ISO639_1_SUPPORTED_LANGS
    @classmethod
    def validate_language(cls, language: str | None) -> str | None: ...
    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType: ...
    @classmethod
    def get_language_token_ids(cls, tokenizer: object) -> list[int]: ...
    @classmethod
    def get_language_detection_prompt(
        cls, audio: np.ndarray, stt_config: SpeechToTextConfig
    ) -> PromptType: ...
    @classmethod
    def parse_language_detection_output(
        cls, token_ids: list[int], tokenizer: object
    ) -> str | None: ...
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig: ...
    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None: ...
    config: Incomplete
    dtype: Incomplete
    model: Incomplete
    proj_out: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
