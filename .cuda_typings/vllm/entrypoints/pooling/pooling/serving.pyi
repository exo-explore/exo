from _typeshed import Incomplete
from fastapi import Request as Request
from typing import Final, Literal
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
)
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as ErrorResponse,
    UsageInfo as UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing as OpenAIServing
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest as IOProcessorRequest,
    IOProcessorResponse as IOProcessorResponse,
    PoolingBytesResponse as PoolingBytesResponse,
    PoolingChatRequest as PoolingChatRequest,
    PoolingCompletionRequest as PoolingCompletionRequest,
    PoolingRequest as PoolingRequest,
    PoolingResponse as PoolingResponse,
    PoolingResponseData as PoolingResponseData,
)
from vllm.entrypoints.pooling.utils import (
    encode_pooling_bytes as encode_pooling_bytes,
    encode_pooling_output_base64 as encode_pooling_output_base64,
    encode_pooling_output_float as encode_pooling_output_float,
)
from vllm.inputs import ProcessorInputs as ProcessorInputs
from vllm.logger import init_logger as init_logger
from vllm.outputs import PoolingRequestOutput as PoolingRequestOutput
from vllm.renderers.inputs.preprocess import prompt_to_seq as prompt_to_seq
from vllm.utils.async_utils import merge_async_iterators as merge_async_iterators
from vllm.utils.serial_utils import (
    EmbedDType as EmbedDType,
    EncodingFormat as EncodingFormat,
    Endianness as Endianness,
)

logger: Incomplete

class OpenAIServingPooling(OpenAIServing):
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
    async def create_pooling(
        self, request: PoolingRequest, raw_request: Request | None = None
    ) -> (
        PoolingResponse | IOProcessorResponse | PoolingBytesResponse | ErrorResponse
    ): ...
    def request_output_to_pooling_json_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["float", "base64"],
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> PoolingResponse: ...
    def request_output_to_pooling_bytes_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["bytes", "bytes_only"],
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> PoolingBytesResponse: ...
    def request_output_to_pooling_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: EncodingFormat,
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> PoolingResponse | PoolingBytesResponse: ...
