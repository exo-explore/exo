import abc
import numpy as np
import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers.feature_extraction_utils import BatchFeature as BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor
from typing import Literal
from vllm.config import (
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.inputs.data import PromptType as PromptType, TokensPrompt as TokensPrompt
from vllm.logger import init_logger as init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
    SupportsTranscription as SupportsTranscription,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM as Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen2_5OmniAudioFeatureInputs as Qwen2_5OmniAudioFeatureInputs,
    Qwen3OmniMoeAudioEncoder as Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeThinkerMultiModalProcessor as Qwen3OmniMoeThinkerMultiModalProcessor,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from vllm.model_executor.models.whisper import (
    ISO639_1_SUPPORTED_LANGS as ISO639_1_SUPPORTED_LANGS,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem as AudioItem,
    ModalityData as ModalityData,
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems as AudioProcessorItems,
    DictEmbeddingItems as DictEmbeddingItems,
    ModalityDataItems as ModalityDataItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config as cached_tokenizer_from_config
from vllm.transformers_utils.configs.qwen3_asr import (
    Qwen3ASRConfig as Qwen3ASRConfig,
    Qwen3ASRThinkerConfig as Qwen3ASRThinkerConfig,
)
from vllm.transformers_utils.processor import (
    cached_processor_from_config as cached_processor_from_config,
)
from vllm.transformers_utils.processors.qwen3_asr import (
    Qwen3ASRProcessor as Qwen3ASRProcessor,
)

logger: Incomplete

class Qwen3ASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> Qwen3ASRProcessor: ...
    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_data_parser(self) -> MultiModalDataParser: ...

class Qwen3ASRDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3ASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Qwen3ASRMultiModalDataParser(MultiModalDataParser): ...
class Qwen3ASRMultiModalProcessor(Qwen3OmniMoeThinkerMultiModalProcessor): ...

class Qwen3ASRForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    SupportsTranscription,
    metaclass=abc.ABCMeta,
):
    supported_languages = ISO639_1_SUPPORTED_LANGS
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    vllm_config: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    quant_config: Incomplete
    audio_tower: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None: ...
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
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
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
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
