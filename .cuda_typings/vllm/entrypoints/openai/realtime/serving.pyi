import asyncio
import numpy as np
from _typeshed import Incomplete
from collections.abc import AsyncGenerator
from functools import cached_property as cached_property
from typing import Literal
from vllm.engine.protocol import (
    EngineClient as EngineClient,
    StreamingInput as StreamingInput,
)
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.engine.serving import OpenAIServing as OpenAIServing
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.inputs.data import PromptType as PromptType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.models.interfaces import SupportsRealtime as SupportsRealtime
from vllm.renderers.inputs.preprocess import parse_model_prompt as parse_model_prompt

logger: Incomplete

class OpenAIServingRealtime(OpenAIServing):
    task_type: Literal["realtime"]
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
    ) -> None: ...
    @cached_property
    def model_cls(self) -> type[SupportsRealtime]: ...
    async def transcribe_realtime(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
    ) -> AsyncGenerator[StreamingInput, None]: ...
