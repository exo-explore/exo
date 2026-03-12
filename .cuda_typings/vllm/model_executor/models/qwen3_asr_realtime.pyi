import abc
import asyncio
import numpy as np
from _typeshed import Incomplete
from collections.abc import AsyncGenerator
from vllm.config import (
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.inputs.data import PromptType as PromptType, TokensPrompt as TokensPrompt
from vllm.logger import init_logger as init_logger
from vllm.model_executor.models.interfaces import SupportsRealtime as SupportsRealtime
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder as Qwen3ASRDummyInputsBuilder,
    Qwen3ASRForConditionalGeneration as Qwen3ASRForConditionalGeneration,
    Qwen3ASRMultiModalProcessor as Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo as Qwen3ASRProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.cache import (
    BaseMultiModalProcessorCache as BaseMultiModalProcessorCache,
    _I,
)
from vllm.multimodal.inputs import (
    MultiModalKwargsOptionalItems as MultiModalKwargsOptionalItems,
)
from vllm.multimodal.parse import MultiModalDataItems as MultiModalDataItems
from vllm.multimodal.processing import BaseDummyInputsBuilder as BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates as MultiModalPromptUpdates,
    PlaceholderFeaturesInfo as PlaceholderFeaturesInfo,
)
from vllm.tokenizers import cached_tokenizer_from_config as cached_tokenizer_from_config
from vllm.transformers_utils.processor import (
    cached_processor_from_config as cached_processor_from_config,
)

logger: Incomplete

class Qwen3ASRRealtimeBuffer:
    def __init__(self, sampling_rate: int, segment_duration_s: float = 5.0) -> None: ...
    def write_audio(self, audio: np.ndarray) -> None: ...
    def read_audio(self) -> np.ndarray | None: ...
    def flush(self) -> np.ndarray | None: ...

class Qwen3ASRRealtimeMultiModalProcessor(Qwen3ASRMultiModalProcessor):
    def __init__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None: ...

class Qwen3ASRRealtimeGeneration(
    Qwen3ASRForConditionalGeneration, SupportsRealtime, metaclass=abc.ABCMeta
):
    realtime_max_tokens: int
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]: ...
    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig: ...
