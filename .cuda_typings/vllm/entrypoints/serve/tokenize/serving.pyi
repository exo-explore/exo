from _typeshed import Incomplete
from dataclasses import dataclass
from fastapi import Request as Request
from typing import Any, Final
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
)
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing as OpenAIServing
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest as DetokenizeRequest,
    DetokenizeResponse as DetokenizeResponse,
    TokenizeChatRequest as TokenizeChatRequest,
    TokenizeRequest as TokenizeRequest,
    TokenizeResponse as TokenizeResponse,
    TokenizerInfoResponse as TokenizerInfoResponse,
)
from vllm.inputs import TokensPrompt as TokensPrompt, token_inputs as token_inputs
from vllm.logger import init_logger as init_logger
from vllm.tokenizers import TokenizerLike as TokenizerLike

logger: Incomplete

class OpenAIServingTokenization(OpenAIServing):
    chat_template: Incomplete
    chat_template_content_format: Final[Incomplete]
    trust_request_chat_template: Incomplete
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
    ) -> None: ...
    async def create_tokenize(
        self, request: TokenizeRequest, raw_request: Request
    ) -> TokenizeResponse | ErrorResponse: ...
    async def create_detokenize(
        self, request: DetokenizeRequest, raw_request: Request
    ) -> DetokenizeResponse | ErrorResponse: ...
    async def get_tokenizer_info(self) -> TokenizerInfoResponse | ErrorResponse: ...

@dataclass
class TokenizerInfo:
    tokenizer: TokenizerLike
    chat_template: str | None
    def to_dict(self) -> dict[str, Any]: ...
