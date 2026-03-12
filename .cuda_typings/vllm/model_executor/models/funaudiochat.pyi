import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property as cached_property
from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor
from transformers.modeling_outputs import BaseModelOutput
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems as AudioProcessorItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
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

class _SinusoidsPositionEmbedding(nn.Module):
    def __init__(
        self, length: int, channels: int, max_timescale: float = 10000.0
    ) -> None: ...

class FunAudioChatAudioAttention(nn.Module):
    embed_dim: Incomplete
    total_num_heads: Incomplete
    dropout: Incomplete
    head_dim: Incomplete
    num_key_value_groups: int
    config: Incomplete
    scaling: Incomplete
    attention_dropout: float
    is_decoder: bool
    is_causal: bool
    qkv_proj: Incomplete
    num_heads: Incomplete
    num_kv_heads: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    attn: Incomplete
    out_proj: Incomplete
    def __init__(self, config: Any) -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...

class FunAudioChatAudioEncoderLayer(nn.Module):
    embed_dim: Incomplete
    self_attn: Incomplete
    self_attn_layer_norm: Incomplete
    dropout: Incomplete
    activation_fn: Incomplete
    activation_dropout: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    final_layer_norm: Incomplete
    def __init__(self, config: Any) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor]: ...

class FunAudioChatAudioEncoder(nn.Module):
    config: Incomplete
    num_mel_bins: Incomplete
    max_source_positions: Incomplete
    embed_scale: Incomplete
    n_window: Incomplete
    conv1: Incomplete
    conv2: Incomplete
    layers: Incomplete
    ln_post: Incomplete
    avg_pooler: Incomplete
    proj: Incomplete
    positional_embedding: Incomplete
    audio_bos_eos_token: Incomplete
    def __init__(self, config: Any) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        aftercnn_lens: torch.Tensor,
        speech_maxlen: int,
        **kwargs: object,
    ) -> BaseModelOutput: ...
    def padded_and_mask_function(
        self,
        tensor_list: Sequence[torch.Tensor],
        tensor_len: torch.Tensor,
        padding_value: float = 0.0,
        padding_side: str = "right",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class FunAudioChatDiscreteEncoder(nn.Module):
    padding_idx: Incomplete
    group_size: Incomplete
    hidden_size: Incomplete
    continuous_features_mode: Incomplete
    embed_tokens: Incomplete
    output_matching: Incomplete
    continual_output_matching: Incomplete
    def __init__(self, config: Any) -> None: ...
    def forward(
        self,
        audio_ids: torch.Tensor,
        continuous_audio_features: torch.Tensor | None = None,
        continuous_audio_output_lengths: torch.Tensor | None = None,
        feature_exist_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class FunAudioChatProcessingInfo(BaseProcessingInfo):
    token_fps: int
    @cached_property
    def feature_extractor(self) -> WhisperFeatureExtractor: ...
    @cached_property
    def speech_tokenizer(self) -> PreTrainedTokenizerFast: ...
    def get_feature_extractor(self) -> WhisperFeatureExtractor: ...
    def get_speech_tokenizer(self) -> PreTrainedTokenizerFast: ...
    def get_data_parser(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_target_channels(self) -> int: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None: ...
    def get_audio_group_size(self) -> int: ...

class FunAudioChatDummyInputsBuilder(
    BaseDummyInputsBuilder[FunAudioChatProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class FunAudioChatMultiModalProcessor(
    BaseMultiModalProcessor[FunAudioChatProcessingInfo]
): ...

class FunAudioChatForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    quant_config: Incomplete
    continuous_audio_tower: Incomplete
    audio_tower: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
