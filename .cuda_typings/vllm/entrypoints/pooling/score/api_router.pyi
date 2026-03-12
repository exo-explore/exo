from _typeshed import Incomplete
from fastapi import Request as Request
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest as RerankRequest,
    RerankResponse as RerankResponse,
    ScoreRequest as ScoreRequest,
    ScoreResponse as ScoreResponse,
)
from vllm.entrypoints.pooling.score.serving import ServingScores as ServingScores
from vllm.entrypoints.utils import (
    load_aware_call as load_aware_call,
    with_cancellation as with_cancellation,
)
from vllm.logger import init_logger as init_logger

router: Incomplete
logger: Incomplete

def score(request: Request) -> ServingScores | None: ...
def rerank(request: Request) -> ServingScores | None: ...
@with_cancellation
@load_aware_call
async def create_score(request: ScoreRequest, raw_request: Request): ...
@with_cancellation
@load_aware_call
async def create_score_v1(request: ScoreRequest, raw_request: Request): ...
@with_cancellation
@load_aware_call
async def do_rerank(request: RerankRequest, raw_request: Request): ...
@with_cancellation
async def do_rerank_v1(request: RerankRequest, raw_request: Request): ...
@with_cancellation
async def do_rerank_v2(request: RerankRequest, raw_request: Request): ...
