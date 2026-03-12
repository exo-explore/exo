import abc
import numpy as np
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsTranscription as SupportsTranscription,
)
from .qwen3 import Qwen3Model as Qwen3Model
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from torch import nn
from transformers import BatchFeature as BatchFeature, Qwen3Config
from typing import Annotated, Literal
from vllm.config import (
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.inputs.data import PromptType as PromptType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
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
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.transformers_utils.processor import (
    cached_processor_from_config as cached_processor_from_config,
)
from vllm.transformers_utils.processors.funasr import (
    FunASRFeatureExtractor as FunASRFeatureExtractor,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete

def sequence_mask(lengths, maxlen=None, dtype=..., device=None): ...

class LayerNorm(torch.nn.LayerNorm):
    dim: Incomplete
    def __init__(self, nout, dim: int = -1) -> None: ...
    def forward(self, x: torch.Tensor): ...

class EncoderLayerSANM(nn.Module):
    self_attn: Incomplete
    feed_forward: Incomplete
    norm1: Incomplete
    norm2: Incomplete
    in_size: Incomplete
    size: Incomplete
    normalize_before: Incomplete
    def __init__(
        self,
        in_size: int,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        normalize_before: bool = True,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache=None,
        mask_shift_chunk=None,
        mask_att_chunk_encoder=None,
    ): ...

class MultiHeadedAttentionSANM(nn.Module):
    d_k: Incomplete
    h: Incomplete
    out_proj: Incomplete
    linear_q_k_v: Incomplete
    attn: Incomplete
    fsmn_block: Incomplete
    pad_fn: Incomplete
    def __init__(
        self,
        n_head: int,
        in_feat: int,
        n_feat: int,
        kernel_size: int,
        sanm_shift: int = 0,
    ) -> None: ...
    def forward_fsmn(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        mask_shift_chunk: torch.Tensor = None,
    ): ...
    def forward_qkv(self, x: torch.Tensor): ...
    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor,
        mask_att_chunk_encoder: torch.Tensor = None,
    ): ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        mask_shift_chunk: torch.Tensor = None,
        mask_att_chunk_encoder: torch.Tensor = None,
    ): ...

class SinusoidalPositionEncoder(torch.nn.Module):
    def __init__(self, d_model: int = 80) -> None: ...
    def encode(
        self,
        positions: torch.Tensor = None,
        depth: int = None,
        dtype: torch.dtype = ...,
    ): ...
    def forward(self, hidden_states: torch.Tensor): ...

class SenseVoiceEncoderSmall(nn.Module):
    embed: Incomplete
    normalize_before: Incomplete
    encoders0: Incomplete
    encoders: Incomplete
    tp_encoders: Incomplete
    after_norm: Incomplete
    tp_norm: Incomplete
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        tp_blocks: int = 0,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        kernel_size: int = 11,
        sanm_shift: int = 0,
        **kwargs,
    ) -> None: ...
    def output_size(self) -> int: ...
    def forward(self, xs_pad: torch.Tensor, ilens: torch.Tensor): ...

class PositionwiseFeedForward(nn.Module):
    w_1: Incomplete
    w_2: Incomplete
    activation: Incomplete
    def __init__(self, idim: int, hidden_units: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class EncoderLayer(nn.Module):
    self_attn: Incomplete
    feed_forward: Incomplete
    norm1: Incomplete
    norm2: Incomplete
    def __init__(
        self, size: int, self_attn: nn.Module, feed_forward: nn.Module
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class FunASRAudioAttention(nn.Module):
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    num_local_heads: Incomplete
    scaling: Incomplete
    qkv: Incomplete
    out_proj: Incomplete
    attn: Incomplete
    def __init__(self, num_heads: int, embed_dim: int, prefix: str = "") -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor: ...

class Transformer(nn.Module):
    k: Incomplete
    encoder_dim: Incomplete
    llm_dim: Incomplete
    linear1: Incomplete
    relu: Incomplete
    linear2: Incomplete
    blocks: Incomplete
    def __init__(
        self,
        downsample_rate: int = 2,
        encoder_dim: int = 1280,
        llm_dim: int = 4096,
        ffn_dim: int = 2048,
        prefix: str = "",
        **kwargs,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor, ilens: int = 0): ...

class FunASRAudioInputs(TensorSchema):
    input_features: Annotated[list[torch.Tensor] | None, None]
    speech_lengths: Annotated[list[torch.Tensor] | None, None]
    fake_token_lengths: Annotated[list[torch.Tensor] | None, None]

class FunASREncoder(nn.Module):
    audio_encoder: Incomplete
    audio_adaptor: Incomplete
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", init_in_fp32: bool = False
    ) -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class FunASRModel(nn.Module):
    encoder: Incomplete
    decoder: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    feat_permute: bool
    def get_encoder_outputs(
        self,
        speech: torch.Tensor | list[torch.Tensor] | None,
        speech_lengths: torch.Tensor | list[torch.Tensor] | None,
    ) -> torch.Tensor | None: ...

class FunASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Qwen3Config: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_feature_extractor(self, **kwargs: object) -> FunASRFeatureExtractor: ...
    def get_data_parser(self) -> MultiModalDataParser: ...
    def get_target_channels(self) -> int: ...

class FunASRDummyInputsBuilder(BaseDummyInputsBuilder[FunASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class FunASRMultiModalProcessor(BaseMultiModalProcessor[FunASRProcessingInfo]): ...

class FunASRForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    supports_transcription_only: bool
    supports_segment_timestamp: bool
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
    lm_head: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
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
