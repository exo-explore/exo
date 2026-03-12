import abc
import numpy as np
import torch
import torch.nn as nn
from .interfaces import (
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsTranscription as SupportsTranscription,
)
from .utils import (
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import BatchFeature as BatchFeature
from typing import Literal
from vllm.config import (
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.inputs.data import PromptType as PromptType, TokensPrompt as TokensPrompt
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models import SupportsPP as SupportsPP
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.whisper import WhisperEncoder as WhisperEncoder
from vllm.model_executor.models.whisper_causal import (
    WhisperCausalEncoder as WhisperCausalEncoder,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
    NestedTensors as NestedTensors,
)
from vllm.multimodal.parse import (
    AudioProcessorItems as AudioProcessorItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder as BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    MultiModalProcessingInfo as MultiModalProcessingInfo,
    PlaceholderFeaturesInfo as PlaceholderFeaturesInfo,
    ProcessorInputs as ProcessorInputs,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    TimingContext as TimingContext,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config as cached_tokenizer_from_config
from vllm.tokenizers.mistral import MistralTokenizer as MistralTokenizer
from vllm.transformers_utils.processors.voxtral import (
    MistralCommonVoxtralProcessor as MistralCommonVoxtralProcessor,
)

logger: Incomplete
ISO639_1_SUPPORTED_LANGS: Incomplete

class VoxtralProcessingInfo(BaseProcessingInfo):
    def get_tokenizer(self) -> MistralTokenizer: ...
    def get_hf_processor(self, **kwargs) -> MistralCommonVoxtralProcessor: ...
    def get_data_parser(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]: ...
    def get_max_audio_tokens(self) -> int: ...
    def get_max_audio_array_len(self) -> int: ...

class VoxtralDummyInputsBuilder(BaseDummyInputsBuilder[VoxtralProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> ProcessorInputs: ...

class VoxtralMultiModalProcessor(BaseMultiModalProcessor[VoxtralProcessingInfo]): ...

class VoxtralForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
    SupportsTranscription,
    metaclass=abc.ABCMeta,
):
    supported_languages = ISO639_1_SUPPORTED_LANGS
    skip_warmup_audio_preprocessing: bool
    packed_modules_mapping: Incomplete
    tokenizer: Incomplete
    config: Incomplete
    downsample_factor: Incomplete
    language_model: Incomplete
    whisper_encoder: Incomplete
    audio_language_adapter: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def embed_multimodal(
        self, **kwargs
    ) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...] | None: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
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
    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def maybe_update_quant_config(
        self, quant_config: QuantizationConfig
    ) -> QuantizationConfig: ...

class AudioLanguageAdapter(nn.Module):
    w_in: Incomplete
    gelu: Incomplete
    w_out: Incomplete
    def __init__(self, hidden_size: int, dim: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class VoxtralEncoderModel(nn.Module):
    packed_modules_mapping: Incomplete
    mistral_remapping: Incomplete
    config: Incomplete
    dtype: torch.dtype
    is_causal: Incomplete
    whisper_encoder: Incomplete
    mel_filters: Incomplete
    def __init__(self, vllm_config: VllmConfig, *, prefix: str = "") -> None: ...
    def compute_whisper_melspec(
        self, audio_waveforms: torch.Tensor
    ) -> torch.Tensor: ...
    @property
    def downsample_factor(self) -> int: ...
    @property
    def chunk_size(self) -> int: ...
    def prepare_inputs_for_conv(
        self, audio_waveforms: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[int]]: ...
    def forward(
        self, input_features: torch.Tensor | list[torch.Tensor]
    ) -> list[torch.Tensor]: ...
    def load_weight(self, weight: tuple[str, torch.Tensor]) -> str: ...
