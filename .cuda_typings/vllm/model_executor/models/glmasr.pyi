import abc
import numpy as np
import torch
import torch.nn as nn
from .glmasr_utils import (
    DEFAULT_CONV_PARAMS as DEFAULT_CONV_PARAMS,
    DEFAULT_MAX_AUDIO_LEN_S as DEFAULT_MAX_AUDIO_LEN_S,
    DEFAULT_MERGE_FACTOR as DEFAULT_MERGE_FACTOR,
)
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
from .whisper import ISO639_1_SUPPORTED_LANGS as ISO639_1_SUPPORTED_LANGS
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers.models.glmasr import GlmAsrConfig, GlmAsrProcessor
from transformers.models.whisper import (
    WhisperFeatureExtractor as WhisperFeatureExtractor,
)
from typing import Annotated, Literal, TypeAlias
from vllm.config import (
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.inputs.data import PromptType as PromptType, TokensPrompt as TokensPrompt
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb as ApplyRotaryEmb,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems as DictEmbeddingItems,
    ModalityData as ModalityData,
    ModalityDataItems as ModalityDataItems,
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
from vllm.tokenizers import cached_tokenizer_from_config as cached_tokenizer_from_config
from vllm.transformers_utils.processor import (
    cached_processor_from_config as cached_processor_from_config,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class GlmAsrEncoderRotaryEmbedding(nn.Module):
    attention_scaling: Incomplete
    dim: Incomplete
    head_dim: Incomplete
    def __init__(self, config) -> None: ...
    def forward(self, seq_len: int) -> torch.Tensor: ...

class GlmAsrEncoderAttention(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    num_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    tp_size: Incomplete
    num_heads_per_rank: Incomplete
    num_kv_heads_per_rank: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_dim: Incomplete
    apply_rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor: ...

class GlmAsrEncoderMLP(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    fc1: Incomplete
    act_fn: Incomplete
    fc2: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GlmAsrEncoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor: ...

class _GlmAsrEncoderOutput:
    last_hidden_state: Incomplete
    def __init__(self, last_hidden_state: torch.Tensor) -> None: ...

class GlmAsrEncoder(nn.Module):
    packed_modules_mapping: Incomplete
    config: Incomplete
    conv1: Incomplete
    conv2: Incomplete
    layers: Incomplete
    norm: Incomplete
    rotary_emb: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(self, input_features: torch.Tensor) -> _GlmAsrEncoderOutput: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class GlmAsrFeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    input_features: Annotated[torch.Tensor | list[torch.Tensor], None]
    feature_attention_mask: Annotated[torch.Tensor | list[torch.Tensor], None]
    chunk_counts: Annotated[torch.Tensor | list[torch.Tensor], None]

class GlmAsrEmbeddingInputs(TensorSchema):
    type: Literal["audio_embeds"]
    audio_embeds: Annotated[list[torch.Tensor], None]

GlmAsrInputs: TypeAlias = GlmAsrFeatureInputs | GlmAsrEmbeddingInputs

class GlmAsrMultiModalProjector(nn.Module):
    linear_1: Incomplete
    act: Incomplete
    linear_2: Incomplete
    def __init__(
        self,
        config: GlmAsrConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor: ...

class GlmAsrMultiModalDataParser(MultiModalDataParser): ...

class GlmAsrProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> GlmAsrConfig: ...
    def get_hf_processor(self, **kwargs: object) -> GlmAsrProcessor: ...
    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor: ...
    def get_data_parser(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...

class GlmAsrDummyInputsBuilder(BaseDummyInputsBuilder[GlmAsrProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class GlmAsrMultiModalProcessor(BaseMultiModalProcessor["GlmAsrProcessingInfo"]): ...

class GlmAsrForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
    SupportsTranscription,
    metaclass=abc.ABCMeta,
):
    supported_languages = ISO639_1_SUPPORTED_LANGS
    packed_modules_mapping: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    quant_config: Incomplete
    audio_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
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
    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig: ...
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
