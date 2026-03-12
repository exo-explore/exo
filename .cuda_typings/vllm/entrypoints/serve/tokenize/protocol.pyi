from _typeshed import Incomplete
from typing import Annotated, Any, TypeAlias
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam as ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel as OpenAIBaseModel
from vllm.renderers import (
    ChatParams as ChatParams,
    TokenizeParams as TokenizeParams,
    merge_kwargs as merge_kwargs,
)

class TokenizeCompletionRequest(OpenAIBaseModel):
    model: str | None
    prompt: str
    add_special_tokens: bool
    return_token_strs: bool | None
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...

class TokenizeChatRequest(OpenAIBaseModel):
    model: str | None
    messages: list[ChatCompletionMessageParam]
    add_generation_prompt: bool
    return_token_strs: bool | None
    continue_final_message: bool
    add_special_tokens: bool
    chat_template: str | None
    chat_template_kwargs: dict[str, Any] | None
    media_io_kwargs: dict[str, dict[str, Any]] | None
    mm_processor_kwargs: dict[str, Any] | None
    tools: list[ChatCompletionToolsParam] | None
    @classmethod
    def check_generation_prompt(cls, data): ...
    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams: ...
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...

TokenizeRequest: TypeAlias = TokenizeCompletionRequest | TokenizeChatRequest

class TokenizeResponse(OpenAIBaseModel):
    count: int
    max_model_len: int
    tokens: list[int]
    token_strs: list[str] | None

class DetokenizeRequest(OpenAIBaseModel):
    model: str | None
    tokens: list[Annotated[int, None]]
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...

class DetokenizeResponse(OpenAIBaseModel):
    prompt: str

class TokenizerInfoResponse(OpenAIBaseModel):
    model_config: Incomplete
    tokenizer_class: str
