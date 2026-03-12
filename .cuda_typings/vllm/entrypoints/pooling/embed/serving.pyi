from _typeshed import Incomplete
from fastapi.responses import JSONResponse as JSONResponse
from typing import TypeAlias
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateConfig as ChatTemplateConfig
from vllm.entrypoints.openai.engine.protocol import UsageInfo as UsageInfo
from vllm.entrypoints.pooling.base.serving import PoolingServing as PoolingServing
from vllm.entrypoints.pooling.embed.io_processor import (
    EmbedIOProcessor as EmbedIOProcessor,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingBytesResponse as EmbeddingBytesResponse,
    EmbeddingRequest as EmbeddingRequest,
    EmbeddingResponse as EmbeddingResponse,
    EmbeddingResponseData as EmbeddingResponseData,
)
from vllm.entrypoints.pooling.typing import PoolingServeContext as PoolingServeContext
from vllm.entrypoints.pooling.utils import (
    encode_pooling_bytes as encode_pooling_bytes,
    encode_pooling_output_base64 as encode_pooling_output_base64,
    encode_pooling_output_float as encode_pooling_output_float,
    get_json_response_cls as get_json_response_cls,
)
from vllm.outputs import PoolingRequestOutput as PoolingRequestOutput
from vllm.renderers import BaseRenderer as BaseRenderer
from vllm.utils.serial_utils import EmbedDType as EmbedDType, Endianness as Endianness

JSONResponseCLS: Incomplete
EmbeddingServeContext: TypeAlias = PoolingServeContext[EmbeddingRequest]

class ServingEmbedding(PoolingServing):
    request_id_prefix: str
    def init_io_processor(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ) -> EmbedIOProcessor: ...
