from _typeshed import Incomplete
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from fastapi import Request as Request
from typing import Any, Generic, TypeAlias, TypeVar
from vllm import PoolingRequestOutput as PoolingRequestOutput
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationChatRequest as ClassificationChatRequest,
    ClassificationCompletionRequest as ClassificationCompletionRequest,
    ClassificationResponse as ClassificationResponse,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingBytesResponse as EmbeddingBytesResponse,
    EmbeddingChatRequest as EmbeddingChatRequest,
    EmbeddingCompletionRequest as EmbeddingCompletionRequest,
    EmbeddingResponse as EmbeddingResponse,
)
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest as IOProcessorRequest,
    PoolingChatRequest as PoolingChatRequest,
    PoolingCompletionRequest as PoolingCompletionRequest,
    PoolingResponse as PoolingResponse,
)
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest as RerankRequest,
    ScoreRequest as ScoreRequest,
    ScoreResponse as ScoreResponse,
)
from vllm.inputs import ProcessorInputs as ProcessorInputs
from vllm.lora.request import LoRARequest as LoRARequest

PoolingCompletionLikeRequest: TypeAlias = (
    EmbeddingCompletionRequest
    | ClassificationCompletionRequest
    | PoolingCompletionRequest
)
PoolingChatLikeRequest: TypeAlias = (
    EmbeddingChatRequest | ClassificationChatRequest | PoolingChatRequest
)
AnyPoolingRequest: TypeAlias = (
    PoolingCompletionLikeRequest
    | PoolingChatLikeRequest
    | IOProcessorRequest
    | RerankRequest
    | ScoreRequest
)
AnyPoolingResponse: TypeAlias = (
    ClassificationResponse
    | EmbeddingResponse
    | EmbeddingBytesResponse
    | PoolingResponse
    | ScoreResponse
)
PoolingRequestT = TypeVar("PoolingRequestT", bound=AnyPoolingRequest)

@dataclass(kw_only=True)
class PoolingServeContext(Generic[PoolingRequestT]):
    request: PoolingRequestT
    raw_request: Request | None = ...
    model_name: str
    request_id: str
    created_time: int = field(default_factory=Incomplete)
    lora_request: LoRARequest | None = ...
    engine_prompts: list[ProcessorInputs] | None = ...
    prompt_request_ids: list[str] | None = ...
    intermediates: Any | None = ...
    result_generator: (
        AsyncGenerator[tuple[int, PoolingRequestOutput], None] | None
    ) = ...
    final_res_batch: list[PoolingRequestOutput] = field(default_factory=list)
    model_config = ...
