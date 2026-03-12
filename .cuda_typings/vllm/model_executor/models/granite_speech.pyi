import abc
import numpy as np
import torch
from .blip2 import Blip2QFormerModel as Blip2QFormerModel
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
    SupportsTranscription as SupportsTranscription,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from torch import nn
from transformers import (
    BatchFeature as BatchFeature,
    PretrainedConfig as PretrainedConfig,
)
from typing import Annotated, Literal
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.inputs.data import PromptType as PromptType, TokensPrompt as TokensPrompt
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
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
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config as cached_tokenizer_from_config
from vllm.transformers_utils.processor import (
    cached_processor_from_config as cached_processor_from_config,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

ISO639_1_SUPPORTED_LANGS: Incomplete

class GraniteSpeechAudioInputs(TensorSchema):
    input_features: Annotated[torch.Tensor, None]
    input_features_mask: Annotated[torch.Tensor, None]
    audio_embed_sizes: Annotated[list[int], None]

class GraniteSpeechMultiModalProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_max_audio_tokens(self): ...
    def get_max_audio_len(self): ...

class GraniteSpeechMultiModalProcessor(
    BaseMultiModalProcessor[GraniteSpeechMultiModalProcessingInfo]
): ...

class GraniteSpeechDummyInputsBuilder(
    BaseDummyInputsBuilder[GraniteSpeechMultiModalProcessingInfo]
):
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...

class GraniteSpeechEncoderProjector(nn.Module):
    hidden_size: Incomplete
    downsample_rate: Incomplete
    window_size: Incomplete
    num_queries: Incomplete
    query: Incomplete
    qformer: Incomplete
    linear: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteSpeechConformerFeedForward(nn.Module):
    pre_norm: Incomplete
    up_proj: Incomplete
    silu: Incomplete
    down_proj: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteSpeechConformerAttention(nn.Module):
    max_pos_emb: Incomplete
    context_size: Incomplete
    num_heads: Incomplete
    dim_head: Incomplete
    scale: Incomplete
    pre_norm: Incomplete
    to_q: Incomplete
    to_kv: Incomplete
    to_out: Incomplete
    rel_pos_emb: Incomplete
    def __init__(self, config: PretrainedConfig, prefix: str = "") -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_dists: torch.Tensor
    ) -> torch.Tensor: ...

class GraniteSpeechConformerDepthWiseConv1d(nn.Module):
    padding: Incomplete
    conv: Incomplete
    def __init__(
        self, chan_in: int, chan_out: int, kernel_size: int, prefix: str = ""
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteSpeechConformerConvModule(nn.Module):
    norm: Incomplete
    up_conv: Incomplete
    glu: Incomplete
    depth_conv: Incomplete
    silu: Incomplete
    batch_norm: Incomplete
    down_conv: Incomplete
    def __init__(self, config: PretrainedConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteSpeechConformerBlock(nn.Module):
    ff1: Incomplete
    attn: Incomplete
    conv: Incomplete
    ff2: Incomplete
    post_norm: Incomplete
    def __init__(self, config: PretrainedConfig, prefix: str = "") -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_dists: torch.Tensor
    ) -> torch.Tensor: ...

class GraniteSpeechCTCEncoder(nn.Module):
    config: Incomplete
    attention_dists: Incomplete
    input_linear: Incomplete
    layers: Incomplete
    out: Incomplete
    out_mid: Incomplete
    softmax: Incomplete
    num_layers: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class GraniteSpeechForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
    SupportsTranscription,
    metaclass=abc.ABCMeta,
):
    supported_languages = ISO639_1_SUPPORTED_LANGS
    packed_modules_mapping: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    quant_config: Incomplete
    cache_config: Incomplete
    language_model: Incomplete
    encoder: Incomplete
    projector: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
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
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None: ...
    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig: ...
