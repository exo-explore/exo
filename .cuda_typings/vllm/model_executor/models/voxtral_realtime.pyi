import abc
import asyncio
import numpy as np
import torch
from _typeshed import Incomplete
from collections.abc import AsyncGenerator, Iterable
from mistral_common.tokens.tokenizers.audio import AudioConfig as AudioConfig
from typing import Literal
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.engine.protocol import StreamingInput as StreamingInput
from vllm.envs import VLLM_ENGINE_ITERATION_TIMEOUT_S as VLLM_ENGINE_ITERATION_TIMEOUT_S
from vllm.inputs.data import PromptType as PromptType, TokensPrompt as TokensPrompt
from vllm.logger import init_logger as init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsRealtime as SupportsRealtime,
)
from vllm.model_executor.models.voxtral import (
    VoxtralDummyInputsBuilder as VoxtralDummyInputsBuilder,
    VoxtralForConditionalGeneration as VoxtralForConditionalGeneration,
    VoxtralMultiModalProcessor as VoxtralMultiModalProcessor,
    VoxtralProcessingInfo as VoxtralProcessingInfo,
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
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config as cached_tokenizer_from_config
from vllm.utils.torch_utils import is_torch_equal_or_newer as is_torch_equal_or_newer

logger: Incomplete

class VoxtralRealtimeMultiModalProcessor(VoxtralMultiModalProcessor):
    def __init__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None: ...

class TimeEmbedding(torch.nn.Module):
    dim: Incomplete
    theta: Incomplete
    def __init__(self, dim: int, theta: float = 10000.0) -> None: ...
    def forward(self, t: torch.Tensor) -> torch.Tensor: ...

class VoxtralRealtimeBuffer:
    def __init__(self, config: AudioConfig, prompt_tokens: list[int]) -> None: ...
    async def append_audio(self, audio_array: np.ndarray | None) -> None: ...
    async def append_tokens(self, tokens: Iterable[int]) -> None: ...
    async def get_input_stream(self) -> AsyncGenerator[StreamingInput]: ...

class VoxtralRealtimeGeneration(
    VoxtralForConditionalGeneration, SupportsRealtime, metaclass=abc.ABCMeta
):
    requires_raw_input_tokens: bool
    skip_warmup_audio_preprocessing: bool
    time_embedding: TimeEmbedding
    n_delay_tokens: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]: ...
    @property
    def audio_config(self): ...
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
    def embed_multimodal(
        self, **kwargs
    ) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...] | None: ...
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
