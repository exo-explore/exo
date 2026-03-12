from typing import Annotated, Any
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
)
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel as OpenAIBaseModel
from vllm.renderers import ChatParams as ChatParams, merge_kwargs as merge_kwargs
from vllm.utils import random_uuid as random_uuid
from vllm.utils.serial_utils import (
    EmbedDType as EmbedDType,
    EncodingFormat as EncodingFormat,
    Endianness as Endianness,
)

class PoolingBasicRequestMixin(OpenAIBaseModel):
    model: str | None
    user: str | None
    truncate_prompt_tokens: Annotated[int, None] | None
    request_id: str
    priority: int
    mm_processor_kwargs: dict[str, Any] | None
    cache_salt: str | None

class CompletionRequestMixin(OpenAIBaseModel):
    input: list[int] | list[list[int]] | str | list[str]
    add_special_tokens: bool

class ChatRequestMixin(OpenAIBaseModel):
    messages: list[ChatCompletionMessageParam]
    add_generation_prompt: bool
    continue_final_message: bool
    add_special_tokens: bool
    chat_template: str | None
    chat_template_kwargs: dict[str, Any] | None
    media_io_kwargs: dict[str, dict[str, Any]] | None
    @classmethod
    def check_generation_prompt(cls, data): ...
    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams: ...

class EncodingRequestMixin(OpenAIBaseModel):
    encoding_format: EncodingFormat
    embed_dtype: EmbedDType
    endianness: Endianness

class EmbedRequestMixin(EncodingRequestMixin):
    dimensions: int | None
    use_activation: bool | None

class ClassifyRequestMixin(OpenAIBaseModel):
    use_activation: bool | None
