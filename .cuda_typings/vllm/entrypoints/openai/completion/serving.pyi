from _typeshed import Incomplete
from collections.abc import AsyncGenerator, AsyncIterator
from fastapi import Request as Request
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.completion.protocol import (
    CompletionLogProbs as CompletionLogProbs,
    CompletionRequest as CompletionRequest,
    CompletionResponse as CompletionResponse,
    CompletionResponseChoice as CompletionResponseChoice,
    CompletionResponseStreamChoice as CompletionResponseStreamChoice,
    CompletionStreamResponse as CompletionStreamResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as ErrorResponse,
    PromptTokenUsageInfo as PromptTokenUsageInfo,
    RequestResponseMetadata as RequestResponseMetadata,
    UsageInfo as UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    GenerationError as GenerationError,
    OpenAIServing as OpenAIServing,
    clamp_prompt_logprobs as clamp_prompt_logprobs,
)
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.utils import (
    get_max_tokens as get_max_tokens,
    should_include_usage as should_include_usage,
)
from vllm.exceptions import VLLMValidationError as VLLMValidationError
from vllm.inputs.data import ProcessorInputs as ProcessorInputs
from vllm.logger import init_logger as init_logger
from vllm.logprobs import Logprob as Logprob
from vllm.outputs import RequestOutput as RequestOutput
from vllm.sampling_params import (
    BeamSearchParams as BeamSearchParams,
    SamplingParams as SamplingParams,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils.async_utils import merge_async_iterators as merge_async_iterators
from vllm.utils.collection_utils import as_list as as_list

logger: Incomplete

class OpenAIServingCompletion(OpenAIServing):
    enable_prompt_tokens_details: Incomplete
    enable_force_include_usage: Incomplete
    default_sampling_params: Incomplete
    override_max_tokens: Incomplete
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
    ) -> None: ...
    async def render_completion_request(
        self, request: CompletionRequest
    ) -> list[ProcessorInputs] | ErrorResponse: ...
    async def create_completion(
        self, request: CompletionRequest, raw_request: Request | None = None
    ) -> AsyncGenerator[str, None] | CompletionResponse | ErrorResponse: ...
    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        engine_prompts: list[ProcessorInputs],
        result_generator: AsyncIterator[tuple[int, RequestOutput]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: TokenizerLike | None,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]: ...
    def request_output_to_completion_response(
        self,
        final_res_batch: list[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
        tokenizer: TokenizerLike | None,
        request_metadata: RequestResponseMetadata,
    ) -> CompletionResponse: ...
