from .io_processor import PoolingIOProcessor as PoolingIOProcessor
from _typeshed import Incomplete
from fastapi import Request as Request
from fastapi.responses import Response as Response
from starlette.datastructures import Headers as Headers
from typing import ClassVar
from vllm import (
    PoolingParams as PoolingParams,
    PoolingRequestOutput as PoolingRequestOutput,
    envs as envs,
)
from vllm.config import ModelConfig as ModelConfig
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateConfig as ChatTemplateConfig,
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
)
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.pooling.typing import (
    AnyPoolingRequest as AnyPoolingRequest,
    PoolingServeContext as PoolingServeContext,
)
from vllm.exceptions import VLLMNotFoundError as VLLMNotFoundError
from vllm.inputs.data import ProcessorInputs as ProcessorInputs
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.renderers.base import BaseRenderer as BaseRenderer
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components as extract_prompt_components,
)
from vllm.tracing import (
    contains_trace_headers as contains_trace_headers,
    extract_trace_headers as extract_trace_headers,
    log_tracing_disabled_warning as log_tracing_disabled_warning,
)
from vllm.utils import random_uuid as random_uuid
from vllm.utils.async_utils import merge_async_iterators as merge_async_iterators

class PoolingServing:
    request_id_prefix: ClassVar[str]
    engine_client: Incomplete
    models: Incomplete
    model_config: Incomplete
    max_model_len: Incomplete
    request_logger: Incomplete
    return_tokens_as_token_ids: Incomplete
    log_error_stack: Incomplete
    chat_template_config: Incomplete
    io_processor: Incomplete
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        log_error_stack: bool = False,
    ) -> None: ...
    def init_io_processor(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ) -> PoolingIOProcessor: ...
    async def __call__(
        self, request: AnyPoolingRequest, raw_request: Request | None = None
    ) -> Response: ...
