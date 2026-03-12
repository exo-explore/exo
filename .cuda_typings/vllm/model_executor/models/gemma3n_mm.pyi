import abc
import numpy as np
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsTranscription as SupportsTranscription,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from torch import nn
from transformers import BatchFeature as BatchFeature
from transformers.models.gemma3n import (
    Gemma3nAudioConfig as Gemma3nAudioConfig,
    Gemma3nAudioFeatureExtractor as Gemma3nAudioFeatureExtractor,
    Gemma3nProcessor,
    Gemma3nTextConfig as Gemma3nTextConfig,
    Gemma3nVisionConfig as Gemma3nVisionConfig,
)
from transformers.models.siglip import (
    SiglipImageProcessorFast as SiglipImageProcessorFast,
)
from typing import Annotated, Literal
from vllm.config import (
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.inputs.data import PromptType as PromptType, TextPrompt as TextPrompt
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear as RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.models.gemma3n import Gemma3nForCausalLM as Gemma3nForCausalLM
from vllm.model_executor.models.gemma3n_audio_utils import (
    adjust_audio_features_to_expected_length as adjust_audio_features_to_expected_length,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.whisper import (
    ISO639_1_SUPPORTED_LANGS as ISO639_1_SUPPORTED_LANGS,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems as ImageProcessorItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder as BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    MultiModalPromptUpdates as MultiModalPromptUpdates,
    MultiModalPromptUpdatesApplyResult as MultiModalPromptUpdatesApplyResult,
    PlaceholderFeaturesInfo as PlaceholderFeaturesInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
    replace_token_matches as replace_token_matches,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

logger: Incomplete
TOKENS_PER_IMAGE: int
TOKENS_PER_AUDIO: int

class Gemma3nImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]

class Gemma3nAudioInputs(TensorSchema):
    type: Literal["audio"]
    input_features_padded: Annotated[torch.Tensor, None]
    input_features_mask: Annotated[torch.Tensor, None]

Gemma3nImageInputs = Gemma3nImagePixelInputs

class Gemma3nProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_feature_extractor(
        self, **kwargs: object
    ) -> Gemma3nAudioFeatureExtractor: ...
    def get_data_parser(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None: ...
    def get_image_repl(
        self, *, image_width: int, image_height: int, processor: Gemma3nProcessor
    ) -> str: ...
    def get_audio_repl(self, *, processor: Gemma3nProcessor) -> str: ...

class Gemma3nDummyInputsBuilder(BaseDummyInputsBuilder[Gemma3nProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Gemma3nMultiModalProcessor(BaseMultiModalProcessor[Gemma3nProcessingInfo]): ...

class Gemma3nMultimodalEmbedder(nn.Module):
    multimodal_hidden_size: Incomplete
    eps: Incomplete
    vocab_offset: Incomplete
    vocab_size: Incomplete
    text_hidden_size: Incomplete
    embedding: Incomplete
    hard_embedding_norm: Incomplete
    soft_embedding_norm: Incomplete
    embedding_projection: Incomplete
    embedding_post_projection_norm: Incomplete
    def __init__(
        self,
        multimodal_config: Gemma3nAudioConfig | Gemma3nVisionConfig,
        text_config: Gemma3nTextConfig,
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class Gemma3nForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsTranscription, metaclass=abc.ABCMeta
):
    supported_languages = ISO639_1_SUPPORTED_LANGS
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    config: Incomplete
    quant_config: Incomplete
    multimodal_config: Incomplete
    vocab_size: Incomplete
    vision_tower: Incomplete
    embed_vision: Incomplete
    audio_tower: Incomplete
    embed_audio: Incomplete
    language_model: Gemma3nForCausalLM
    per_layer_embeddings: Incomplete
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
    ) -> IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType: ...
    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig: ...
