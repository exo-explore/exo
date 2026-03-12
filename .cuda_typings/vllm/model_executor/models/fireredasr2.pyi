import abc
import numpy as np
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsTranscription as SupportsTranscription,
)
from .qwen2 import Qwen2ForCausalLM as Qwen2ForCausalLM
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from torch import nn
from transformers import BatchFeature as BatchFeature, Qwen2Config
from typing import Annotated, Literal
from vllm.config import (
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.inputs.data import PromptType as PromptType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
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
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.transformers_utils.processor import (
    cached_processor_from_config as cached_processor_from_config,
)
from vllm.transformers_utils.processors.fireredasr2 import (
    FireRedASR2FeatureExtractor as FireRedASR2FeatureExtractor,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete

class FireRedASR2AudioInputs(TensorSchema):
    input_features: Annotated[list[torch.Tensor] | None, None]
    speech_lengths: Annotated[list[torch.Tensor] | None, None]
    fake_token_lengths: Annotated[list[torch.Tensor] | None, None]

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Conv2dSubsampling(nn.Module):
    conv: Incomplete
    out: Incomplete
    subsampling: int
    context: Incomplete
    def __init__(self, idim: int, d_model: int, out_channels: int = 32) -> None: ...
    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class RelPositionalEncoding(nn.Module):
    pe: Incomplete
    def __init__(self, d_model: int, max_len: int = 5000) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class ConformerFeedForward(nn.Module):
    pre_layer_norm: Incomplete
    linear_expand: Incomplete
    nonlinear: Incomplete
    linear_project: Incomplete
    def __init__(self, d_model: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class EncoderMultiHeadAttention(nn.Module):
    n_head: Incomplete
    d_k: Incomplete
    d_v: Incomplete
    w_qs: Incomplete
    w_ks: Incomplete
    w_vs: Incomplete
    layer_norm_q: Incomplete
    layer_norm_k: Incomplete
    layer_norm_v: Incomplete
    fc: Incomplete
    def __init__(self, n_head: int, d_model: int) -> None: ...
    def forward_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def forward_output(
        self, output: torch.Tensor, residual: torch.Tensor, sz_b: int, len_q: int
    ) -> torch.Tensor: ...
    def forward_attention(
        self, attn: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    scale: Incomplete
    linear_pos: Incomplete
    pos_bias_u: Incomplete
    pos_bias_v: Incomplete
    def __init__(self, n_head: int, d_model: int) -> None: ...
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class ConformerConvolution(nn.Module):
    pre_layer_norm: Incomplete
    pointwise_conv1: Incomplete
    padding: Incomplete
    depthwise_conv: Incomplete
    batch_norm: Incomplete
    swish: Incomplete
    pointwise_conv2: Incomplete
    def __init__(self, d_model: int, kernel_size: int = 33) -> None: ...
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class RelPosEmbConformerBlock(nn.Module):
    ffn1: Incomplete
    mhsa: Incomplete
    conv: Incomplete
    ffn2: Incomplete
    layer_norm: Incomplete
    def __init__(self, d_model, n_head, kernel_size: int = 33) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        slf_attn_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class ConformerEncoder(nn.Module):
    odim: Incomplete
    input_preprocessor: Incomplete
    positional_encoding: Incomplete
    layer_stack: Incomplete
    def __init__(
        self,
        idim: int,
        n_layers_enc: int,
        n_head: int,
        d_model: int,
        kernel_size: int = 33,
        pe_maxlen: int = 5000,
    ) -> None: ...
    def forward(
        self, padded_input: torch.Tensor, input_lengths: torch.Tensor, pad: bool = True
    ): ...
    def padding_position_is_0(
        self, padded_input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor: ...

class FireRedASR2Adapter(nn.Module):
    ds: Incomplete
    linear1: Incomplete
    relu: Incomplete
    linear2: Incomplete
    def __init__(
        self, encoder_dim: int, llm_dim: int, downsample_rate: int = 2
    ) -> None: ...
    def forward(self, x, x_lens): ...

class FireRedASR2Encoder(nn.Module):
    audio_encoder: Incomplete
    def __init__(self, *, vllm_config: VllmConfig) -> None: ...

class FireRedASR2Model(nn.Module):
    encoder: Incomplete
    encoder_projector: Incomplete
    decoder: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def get_encoder_outputs(
        self,
        speech: torch.Tensor | list[torch.Tensor] | None,
        speech_lengths: torch.Tensor | list[torch.Tensor] | None,
    ) -> torch.Tensor | None: ...

class FireRedASR2ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Qwen2Config: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_feature_extractor(
        self, **kwargs: object
    ) -> FireRedASR2FeatureExtractor: ...
    def get_data_parser(self) -> MultiModalDataParser: ...
    def get_target_channels(self) -> int: ...

class FireRedASR2DummyInputsBuilder(BaseDummyInputsBuilder[FireRedASR2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class FireRedASR2MultiModalProcessor(
    BaseMultiModalProcessor[FireRedASR2ProcessingInfo]
): ...

class FireRedASR2ForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
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
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
