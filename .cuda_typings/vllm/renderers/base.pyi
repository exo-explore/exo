import abc
from .embed_utils import safe_load_prompt_embeds as safe_load_prompt_embeds
from .inputs import (
    DictPrompt as DictPrompt,
    EncoderDecoderDictPrompt as EncoderDecoderDictPrompt,
    EncoderDecoderTokPrompt as EncoderDecoderTokPrompt,
    SingletonDictPrompt as SingletonDictPrompt,
    SingletonTokPrompt as SingletonTokPrompt,
    TokPrompt as TokPrompt,
)
from .inputs.preprocess import extract_target_prompt as extract_target_prompt
from .params import ChatParams as ChatParams, TokenizeParams as TokenizeParams
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property as cached_property
from typing import Any, Generic
from vllm.config import VllmConfig as VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
    ConversationMessage as ConversationMessage,
)
from vllm.inputs import (
    EmbedsInputs as EmbedsInputs,
    EmbedsPrompt as EmbedsPrompt,
    EncoderDecoderInputs as EncoderDecoderInputs,
    ProcessorInputs as ProcessorInputs,
    SingletonInputs as SingletonInputs,
    TextPrompt as TextPrompt,
    TokenInputs as TokenInputs,
    TokensPrompt as TokensPrompt,
)
from vllm.inputs.data import (
    build_enc_dec_inputs as build_enc_dec_inputs,
    embeds_inputs as embeds_inputs,
    token_inputs as token_inputs,
)
from vllm.logger import init_logger as init_logger
from vllm.multimodal.cache import (
    BaseMultiModalProcessorCache as BaseMultiModalProcessorCache,
)
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalInputs as MultiModalInputs,
    MultiModalUUIDDict as MultiModalUUIDDict,
)
from vllm.multimodal.parse import (
    MultiModalDataItems as MultiModalDataItems,
    MultiModalUUIDItems as MultiModalUUIDItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils.async_utils import AsyncMicrobatchTokenizer as AsyncMicrobatchTokenizer
from vllm.utils.counter import AtomicCounter as AtomicCounter
from vllm.utils.torch_utils import (
    set_default_torch_num_threads as set_default_torch_num_threads,
)
from vllm.v1.metrics.stats import MultiModalCacheStats as MultiModalCacheStats

logger: Incomplete

class BaseRenderer(ABC, Generic[_T], metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def from_config(
        cls, config: VllmConfig, tokenizer_kwargs: dict[str, Any]
    ) -> BaseRenderer: ...
    config: Incomplete
    model_config: Incomplete
    api_process_rank: Incomplete
    tokenizer: Incomplete
    mm_processor: BaseMultiModalProcessor | None
    def __init__(self, config: VllmConfig, tokenizer: _T | None) -> None: ...
    def get_tokenizer(self) -> _T: ...
    def get_async_tokenizer(self) -> AsyncMicrobatchTokenizer: ...
    def get_mm_processor(self) -> BaseMultiModalProcessor: ...
    @property
    def mm_processor_cache(self) -> BaseMultiModalProcessorCache | None: ...
    def stat_mm_cache(self) -> MultiModalCacheStats | None: ...
    def update_mm_cache_stats(self) -> None: ...
    def clear_mm_cache(self) -> None: ...
    def warmup(self, chat_params: ChatParams) -> None: ...
    def shutdown(self) -> None: ...
    def get_bos_token_id(self) -> int | None: ...
    def get_eos_token_id(self) -> int | None: ...
    def get_dec_start_token_id(self) -> int: ...
    @cached_property
    def default_cmpl_tok_params(self) -> TokenizeParams: ...
    @cached_property
    def default_chat_tok_params(self) -> TokenizeParams: ...
    def render_prompt(self, prompt: DictPrompt | bytes) -> DictPrompt: ...
    def render_prompts(
        self, prompts: Sequence[DictPrompt | bytes]
    ) -> list[DictPrompt]: ...
    async def render_prompts_async(
        self, prompts: Sequence[DictPrompt | bytes]
    ) -> list[DictPrompt]: ...
    @abstractmethod
    def render_messages(
        self, messages: list["ChatCompletionMessageParam"], params: ChatParams
    ) -> tuple[list["ConversationMessage"], DictPrompt]: ...
    async def render_messages_async(
        self, messages: list["ChatCompletionMessageParam"], params: ChatParams
    ) -> tuple[list["ConversationMessage"], DictPrompt]: ...
    def tokenize_prompt(
        self, prompt: DictPrompt, params: TokenizeParams
    ) -> TokPrompt: ...
    def tokenize_prompts(
        self, prompts: Sequence[DictPrompt], params: TokenizeParams
    ) -> list[TokPrompt]: ...
    async def tokenize_prompt_async(
        self, prompt: DictPrompt, params: TokenizeParams
    ) -> TokPrompt: ...
    async def tokenize_prompts_async(
        self, prompts: Sequence[DictPrompt], params: TokenizeParams
    ) -> list[TokPrompt]: ...
    def process_for_engine(
        self, prompt: TokPrompt, arrival_time: float
    ) -> ProcessorInputs: ...
    def render_cmpl(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ): ...
    async def render_cmpl_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ): ...
    def render_chat(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ): ...
    async def render_chat_async(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ): ...
