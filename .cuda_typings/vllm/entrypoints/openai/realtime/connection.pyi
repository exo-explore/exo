import asyncio
import numpy as np
from _typeshed import Incomplete
from collections.abc import AsyncGenerator
from fastapi import WebSocket as WebSocket
from vllm import envs as envs
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as ErrorResponse,
    UsageInfo as UsageInfo,
)
from vllm.entrypoints.openai.realtime.protocol import (
    ErrorEvent as ErrorEvent,
    InputAudioBufferAppend as InputAudioBufferAppend,
    InputAudioBufferCommit as InputAudioBufferCommit,
    SessionCreated as SessionCreated,
    TranscriptionDelta as TranscriptionDelta,
    TranscriptionDone as TranscriptionDone,
)
from vllm.entrypoints.openai.realtime.serving import (
    OpenAIServingRealtime as OpenAIServingRealtime,
)
from vllm.exceptions import VLLMValidationError as VLLMValidationError
from vllm.logger import init_logger as init_logger

logger: Incomplete

class RealtimeConnection:
    websocket: Incomplete
    connection_id: Incomplete
    serving: Incomplete
    audio_queue: asyncio.Queue[np.ndarray | None]
    generation_task: asyncio.Task | None
    def __init__(
        self, websocket: WebSocket, serving: OpenAIServingRealtime
    ) -> None: ...
    async def handle_connection(self) -> None: ...
    async def handle_event(self, event: dict): ...
    async def audio_stream_generator(self) -> AsyncGenerator[np.ndarray, None]: ...
    async def start_generation(self) -> None: ...
    async def send(
        self, event: SessionCreated | TranscriptionDelta | TranscriptionDone
    ): ...
    async def send_error(self, message: str, code: str | None = None): ...
    async def cleanup(self) -> None: ...
