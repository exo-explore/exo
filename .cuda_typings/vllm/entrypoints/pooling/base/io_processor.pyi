from _typeshed import Incomplete
from collections.abc import Callable as Callable, Sequence
from typing import Any, Final
from vllm import PoolingRequestOutput as PoolingRequestOutput, PromptType as PromptType
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
    ChatTemplateConfig as ChatTemplateConfig,
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
    ConversationMessage as ConversationMessage,
)
from vllm.entrypoints.openai.engine.serving import (
    RendererChatRequest as RendererChatRequest,
    RendererRequest as RendererRequest,
)
from vllm.entrypoints.pooling.typing import (
    PoolingChatLikeRequest as PoolingChatLikeRequest,
    PoolingCompletionLikeRequest as PoolingCompletionLikeRequest,
    PoolingServeContext as PoolingServeContext,
)
from vllm.inputs.data import (
    ProcessorInputs as ProcessorInputs,
    SingletonPrompt as SingletonPrompt,
)
from vllm.renderers import BaseRenderer as BaseRenderer, merge_kwargs as merge_kwargs
from vllm.renderers.inputs.preprocess import (
    parse_model_prompt as parse_model_prompt,
    prompt_to_seq as prompt_to_seq,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers import ToolParser as ToolParser
from vllm.utils.mistral import is_mistral_tokenizer as is_mistral_tokenizer

class PoolingIOProcessor:
    name: str
    model_config: Incomplete
    renderer: Incomplete
    chat_template: Incomplete
    chat_template_content_format: Final[Incomplete]
    trust_request_chat_template: Incomplete
    def __init__(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ) -> None: ...
    def create_pooling_params(self, request): ...
    def pre_process_online(self, ctx: PoolingServeContext): ...
    async def pre_process_online_async(self, ctx: PoolingServeContext): ...
    def post_process_online(self, ctx: PoolingServeContext): ...
    async def post_process_online_async(self, ctx: PoolingServeContext): ...
    def pre_process_offline(
        self,
        prompts: PromptType | Sequence[PromptType],
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[ProcessorInputs]: ...
    async def pre_process_offline_async(self, *args, **kwargs): ...
    def post_process_offline(
        self, outputs: list[PoolingRequestOutput]
    ) -> list[PoolingRequestOutput]: ...
    async def post_process_offline_async(
        self, outputs: list[PoolingRequestOutput]
    ) -> list[PoolingRequestOutput]: ...
