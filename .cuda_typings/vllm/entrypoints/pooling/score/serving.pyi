from _typeshed import Incomplete
from fastapi import Request as Request
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as ErrorResponse,
    UsageInfo as UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing as OpenAIServing
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.pooling.score.protocol import (
    RerankDocument as RerankDocument,
    RerankRequest as RerankRequest,
    RerankResponse as RerankResponse,
    RerankResult as RerankResult,
    RerankUsage as RerankUsage,
    ScoreRequest as ScoreRequest,
    ScoreResponse as ScoreResponse,
    ScoreResponseData as ScoreResponseData,
)
from vllm.entrypoints.pooling.score.utils import (
    ScoreData as ScoreData,
    ScoreInputs as ScoreInputs,
    compress_token_type_ids as compress_token_type_ids,
    get_score_prompt as get_score_prompt,
    parse_score_data_single as parse_score_data_single,
    validate_score_input as validate_score_input,
)
from vllm.inputs.data import (
    ProcessorInputs as ProcessorInputs,
    TokensPrompt as TokensPrompt,
    token_inputs as token_inputs,
)
from vllm.logger import init_logger as init_logger
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.outputs import (
    PoolingRequestOutput as PoolingRequestOutput,
    ScoringRequestOutput as ScoringRequestOutput,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils.async_utils import (
    make_async as make_async,
    merge_async_iterators as merge_async_iterators,
)
from vllm.utils.mistral import is_mistral_tokenizer as is_mistral_tokenizer
from vllm.v1.pool.late_interaction import (
    build_late_interaction_doc_params as build_late_interaction_doc_params,
    build_late_interaction_query_params as build_late_interaction_query_params,
)

logger: Incomplete

class ServingScores(OpenAIServing):
    score_template: Incomplete
    score_type: Incomplete
    architecture: Incomplete
    is_multimodal_model: Incomplete
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        score_template: str | None = None,
        log_error_stack: bool = False,
    ) -> None: ...
    async def create_score(
        self, request: ScoreRequest, raw_request: Request | None = None
    ) -> ScoreResponse | ErrorResponse: ...
    async def do_rerank(
        self, request: RerankRequest, raw_request: Request | None = None
    ) -> RerankResponse | ErrorResponse: ...
    def request_output_to_score_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> ScoreResponse: ...
    def request_output_to_rerank_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        model_name: str,
        documents: ScoreInputs,
        top_n: int,
    ) -> RerankResponse: ...
