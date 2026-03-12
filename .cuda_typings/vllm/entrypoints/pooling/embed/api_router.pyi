from _typeshed import Incomplete
from fastapi import Request as Request
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest as EmbeddingRequest
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding as ServingEmbedding
from vllm.entrypoints.utils import (
    load_aware_call as load_aware_call,
    with_cancellation as with_cancellation,
)

router: Incomplete

def embedding(request: Request) -> ServingEmbedding | None: ...
@with_cancellation
@load_aware_call
async def create_embedding(request: EmbeddingRequest, raw_request: Request): ...
