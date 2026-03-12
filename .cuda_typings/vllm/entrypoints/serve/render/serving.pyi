from _typeshed import Incomplete
from collections.abc import Callable as Callable
from http import HTTPStatus
from typing import Any
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
    ConversationMessage as ConversationMessage,
)
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest as CompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo as ErrorInfo,
    ErrorResponse as ErrorResponse,
    ModelCard as ModelCard,
    ModelList as ModelList,
    ModelPermission as ModelPermission,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_developer_message as get_developer_message,
    get_system_message as get_system_message,
    parse_chat_inputs_to_harmony_messages as parse_chat_inputs_to_harmony_messages,
    render_for_completion as render_for_completion,
)
from vllm.entrypoints.utils import sanitize_message as sanitize_message
from vllm.inputs.data import (
    ProcessorInputs as ProcessorInputs,
    PromptType as PromptType,
    SingletonPrompt as SingletonPrompt,
    TokensPrompt as TokensPrompt,
)
from vllm.logger import init_logger as init_logger
from vllm.parser import ParserManager as ParserManager
from vllm.renderers import BaseRenderer as BaseRenderer, merge_kwargs as merge_kwargs
from vllm.renderers.inputs.preprocess import (
    parse_model_prompt as parse_model_prompt,
    prompt_to_seq as prompt_to_seq,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers import ToolParser as ToolParser
from vllm.utils.mistral import is_mistral_tokenizer as is_mistral_tokenizer

logger: Incomplete

class OpenAIServingRender:
    model_config: Incomplete
    renderer: Incomplete
    io_processor: Incomplete
    served_model_names: Incomplete
    request_logger: Incomplete
    chat_template: Incomplete
    chat_template_content_format: ChatTemplateContentFormatOption
    trust_request_chat_template: Incomplete
    enable_auto_tools: Incomplete
    exclude_tools_when_tool_choice_none: Incomplete
    tool_parser: Callable[[TokenizerLike], ToolParser] | None
    default_chat_template_kwargs: dict[str, Any]
    log_error_stack: Incomplete
    use_harmony: Incomplete
    supports_browsing: bool
    supports_code_interpreter: bool
    def __init__(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        io_processor: Any,
        served_model_names: list[str],
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        log_error_stack: bool = False,
    ) -> None: ...
    async def render_chat_request(
        self, request: ChatCompletionRequest
    ) -> tuple[list[ConversationMessage], list[ProcessorInputs]] | ErrorResponse: ...
    async def render_completion_request(
        self, request: CompletionRequest
    ) -> list[ProcessorInputs] | ErrorResponse: ...
    async def show_available_models(self) -> ModelList: ...
    def create_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = ...,
        param: str | None = None,
    ) -> ErrorResponse: ...
