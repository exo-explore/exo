from _typeshed import Incomplete
from collections.abc import AsyncGenerator
from fastapi import Request as Request
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb as ChatCompletionLogProb,
    ChatCompletionLogProbs as ChatCompletionLogProbs,
    ChatCompletionLogProbsContent as ChatCompletionLogProbsContent,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as ErrorResponse,
    PromptTokenUsageInfo as PromptTokenUsageInfo,
    RequestResponseMetadata as RequestResponseMetadata,
    UsageInfo as UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    OpenAIServing as OpenAIServing,
    clamp_prompt_logprobs as clamp_prompt_logprobs,
)
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.serve.disagg.protocol import (
    GenerateRequest as GenerateRequest,
    GenerateResponse as GenerateResponse,
    GenerateResponseChoice as GenerateResponseChoice,
)
from vllm.logger import init_logger as init_logger
from vllm.logprobs import Logprob as Logprob
from vllm.outputs import RequestOutput as RequestOutput
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.utils.collection_utils import as_list as as_list

logger: Incomplete

class ServingTokens(OpenAIServing):
    enable_prompt_tokens_details: Incomplete
    enable_log_outputs: Incomplete
    force_no_detokenize: Incomplete
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        force_no_detokenize: bool = False,
        return_tokens_as_token_ids: bool = False,
        enable_prompt_tokens_details: bool = False,
        enable_log_outputs: bool = False,
    ) -> None: ...
    async def serve_tokens(
        self, request: GenerateRequest, raw_request: Request | None = None
    ) -> GenerateResponse | ErrorResponse: ...
    async def serve_tokens_full_generator(
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | GenerateResponse: ...
