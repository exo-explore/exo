from .base import BaseRenderer as BaseRenderer
from .inputs import DictPrompt as DictPrompt
from .inputs.preprocess import parse_dec_only_prompt as parse_dec_only_prompt
from .params import ChatParams as ChatParams
from _typeshed import Incomplete
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
    ConversationMessage as ConversationMessage,
    parse_chat_messages as parse_chat_messages,
    parse_chat_messages_async as parse_chat_messages_async,
)
from vllm.logger import init_logger as init_logger
from vllm.tokenizers import cached_get_tokenizer as cached_get_tokenizer
from vllm.tokenizers.grok2 import Grok2Tokenizer as Grok2Tokenizer

logger: Incomplete

class Grok2Renderer(BaseRenderer[Grok2Tokenizer]):
    @classmethod
    def from_config(
        cls, config: VllmConfig, tokenizer_kwargs: dict[str, Any]
    ) -> Grok2Renderer: ...
    def render_messages(
        self, messages: list[ChatCompletionMessageParam], params: ChatParams
    ) -> tuple[list[ConversationMessage], DictPrompt]: ...
    async def render_messages_async(
        self, messages: list[ChatCompletionMessageParam], params: ChatParams
    ) -> tuple[list[ConversationMessage], DictPrompt]: ...
