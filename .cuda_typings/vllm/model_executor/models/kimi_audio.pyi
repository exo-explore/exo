import abc
import numpy as np
import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import BatchFeature as BatchFeature
from typing import Any, ClassVar, Literal
from vllm.config import (
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.inputs.data import PromptType as PromptType, TokensPrompt as TokensPrompt
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
    SupportsTranscription as SupportsTranscription,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from vllm.model_executor.models.whisper import WhisperEncoder as WhisperEncoder
from vllm.model_executor.models.whisper_utils import (
    ISO639_1_SUPPORTED_LANGS as ISO639_1_SUPPORTED_LANGS,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig as MultiModalFieldConfig
from vllm.multimodal.parse import (
    AudioItem as AudioItem,
    DictEmbeddingItems as DictEmbeddingItems,
    ModalityData as ModalityData,
    ModalityDataItems as ModalityDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
)
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import cached_get_tokenizer as cached_get_tokenizer
from vllm.tokenizers.kimi_audio import KimiAudioTokenizer as KimiAudioTokenizer
from vllm.transformers_utils.processor import (
    cached_feature_extractor_from_config as cached_feature_extractor_from_config,
)
from vllm.transformers_utils.processors.kimi_audio import (
    KimiAudioProcessor as KimiAudioProcessor,
)
from vllm.v1.sample.metadata import SamplingMetadata as SamplingMetadata

KIMIA_WHISPER_SUBFOLDER: str

class KimiAudioWhisperEncoder(WhisperEncoder):
    packed_modules_mapping: Incomplete
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", init_in_fp32: bool = False
    ) -> None: ...

class KimiAudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> KimiAudioProcessor: ...
    def get_feature_extractor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_data_parser(self) -> KimiAudioMultiModalDataParser: ...

class KimiAudioDummyInputsBuilder(BaseDummyInputsBuilder[KimiAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> list[int]: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

class KimiAudioMultiModalDataParser(MultiModalDataParser):
    def __init__(self, **kwargs) -> None: ...

class KimiAudioMultiModalProcessor(
    BaseMultiModalProcessor[KimiAudioProcessingInfo]
): ...

class KimiAudioMultiModalProjector(nn.Module):
    whisper_dim: Incomplete
    llm_dim: Incomplete
    vq_adaptor_layers_0: Incomplete
    vq_adaptor_layers_3: Incomplete
    vq_adaptor_layers_4: Incomplete
    def __init__(
        self, whisper_dim: int = 5120, llm_dim: int = 3584, prefix: str = ""
    ) -> None: ...
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor: ...

class KimiAudioForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
    metaclass=abc.ABCMeta,
):
    supported_languages: ClassVar[Mapping[str, str]]
    supports_transcription: ClassVar[Literal[True]]
    hf_to_vllm_mapper: Incomplete
    AUDIO_PLACEHOLDER: str
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    quant_config: Incomplete
    multimodal_config: Incomplete
    model_path: Incomplete
    audio_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor] | None: ...
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: tuple[torch.Tensor, ...] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None: ...
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
    @classmethod
    def post_process_output(cls, text: str) -> str: ...
