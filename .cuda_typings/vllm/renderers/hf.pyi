import jinja2.ext
import jinja2.nodes
import jinja2.parser
from .base import BaseRenderer as BaseRenderer
from .inputs import DictPrompt as DictPrompt
from .inputs.preprocess import parse_dec_only_prompt as parse_dec_only_prompt
from .params import ChatParams as ChatParams
from _typeshed import Incomplete
from collections.abc import Set as Set
from typing import Any, Literal, overload
from vllm.config import ModelConfig as ModelConfig, VllmConfig as VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
    ChatTemplateContentFormat as ChatTemplateContentFormat,
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
    ChatTemplateResolutionError as ChatTemplateResolutionError,
    ConversationMessage as ConversationMessage,
    load_chat_template as load_chat_template,
    parse_chat_messages as parse_chat_messages,
    parse_chat_messages_async as parse_chat_messages_async,
)
from vllm.logger import init_logger as init_logger
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalUUIDDict as MultiModalUUIDDict,
)
from vllm.tokenizers import cached_get_tokenizer as cached_get_tokenizer
from vllm.tokenizers.hf import (
    CachedHfTokenizer as CachedHfTokenizer,
    HfTokenizer as HfTokenizer,
)
from vllm.transformers_utils.chat_templates import (
    get_chat_template_fallback_path as get_chat_template_fallback_path,
)
from vllm.transformers_utils.processor import (
    cached_get_processor as cached_get_processor,
)
from vllm.utils.func_utils import supports_kw as supports_kw

logger: Incomplete

def resolve_chat_template(
    tokenizer: HfTokenizer,
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    *,
    model_config: ModelConfig,
) -> str | None: ...
def resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    given_format: ChatTemplateContentFormatOption,
    tokenizer: HfTokenizer,
    *,
    model_config: ModelConfig,
) -> ChatTemplateContentFormat: ...

class AssistantTracker(jinja2.ext.Extension):
    tags: Incomplete
    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.Node: ...

def resolve_chat_template_kwargs(
    tokenizer: HfTokenizer,
    chat_template: str,
    chat_template_kwargs: dict[str, Any],
    raise_on_unexpected: bool = True,
) -> dict[str, Any]: ...
@overload
def safe_apply_chat_template(
    model_config: ModelConfig,
    tokenizer: HfTokenizer,
    conversation: list[ConversationMessage],
    *,
    tools: list[dict[str, Any]] | None = ...,
    chat_template: str | None = ...,
    tokenize: Literal[True] = ...,
    **kwargs,
) -> list[int]: ...
@overload
def safe_apply_chat_template(
    model_config: ModelConfig,
    tokenizer: HfTokenizer,
    conversation: list[ConversationMessage],
    *,
    tools: list[dict[str, Any]] | None = ...,
    chat_template: str | None = ...,
    tokenize: Literal[False] = ...,
    **kwargs,
) -> str: ...
def rebuild_mm_uuids_from_mm_data(
    mm_uuids: MultiModalUUIDDict, mm_data: MultiModalDataDict
) -> MultiModalUUIDDict: ...
def build_video_prompts_from_mm_data(mm_data: MultiModalDataDict) -> list[str]: ...
def replace_vision_chunk_video_placeholder(
    prompt_raw: str | list[int],
    mm_data: MultiModalDataDict,
    video_placeholder: str | None,
) -> str | list[int]: ...

class HfRenderer(BaseRenderer[HfTokenizer]):
    @classmethod
    def from_config(
        cls, config: VllmConfig, tokenizer_kwargs: dict[str, Any]
    ) -> HfRenderer: ...
    use_unified_vision_chunk: Incomplete
    def __init__(self, config: VllmConfig, tokenizer: HfTokenizer | None) -> None: ...
    def render_messages(
        self, messages: list[ChatCompletionMessageParam], params: ChatParams
    ) -> tuple[list[ConversationMessage], DictPrompt]: ...
    async def render_messages_async(
        self, messages: list[ChatCompletionMessageParam], params: ChatParams
    ) -> tuple[list[ConversationMessage], DictPrompt]: ...
