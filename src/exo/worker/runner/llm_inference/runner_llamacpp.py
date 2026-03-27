"""LlamaCpp runner — inference via llama-cpp-python (ROCm / Vulkan / CPU).

This runner implements the same task-state-machine as the MLX runner so the
rest of the system (master, worker, plan loop) is oblivious to which backend
is active.  Communication with the parent ``RunnerSupervisor`` still happens
over the same ``MpSender`` / ``MpReceiver`` multiprocessing channels.

Distributed inference uses llama.cpp's built-in RPC protocol:
- **Rank 1+**: spawns a ``llama-rpc-server`` subprocess and reports
  ``RunnerConnected`` once it is listening.
- **Rank 0**:  waits for all peers to be ``RunnerConnected``, then calls
  ``Llama(rpc_servers=..., n_gpu_layers=...)`` to load the model and
  split GPU layers across the cluster.
"""

import contextlib
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from anyio import WouldBlock

from exo.shared.types.chunks import ErrorChunk, TokenChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance, LlamaCppRpcInstance
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.llamacpp.utils_llamacpp import (
    find_gguf_model_path,
    load_llamacpp_model,
    start_rpc_server,
)
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from llama_cpp import Llama


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Shutdown = "Shutdown"


class LlamaCppRunner:
    """Inference runner backed by llama-cpp-python.

    The public interface (``main()``, task state machine) is intentionally
    identical to the MLX ``Runner`` so the ``RunnerSupervisor`` and planner
    can treat both runners the same way.
    """

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance

        assert isinstance(bound_instance.instance, LlamaCppRpcInstance), (
            "LlamaCppRunner requires a LlamaCppRpcInstance"
        )
        self.llama_instance: LlamaCppRpcInstance = bound_instance.instance

        self.runner_id = bound_instance.bound_runner_id
        self.shard = bound_instance.bound_shard
        self.model_id: ModelId = self.shard.model_card.model_id
        self.device_rank: int = self.shard.device_rank

        self._llm: Llama | None = None
        self._rpc_server_proc: subprocess.Popen[bytes] | None = None
        self._model_path: Path | None = None
        self._cancelled_tasks: set[TaskId] = set()

        self.setup_start_time = time.time()
        self.seen: set[TaskId] = set()
        self.active_tasks: dict[TaskId, TextGeneration] = {}

        logger.info(f"LlamaCppRunner created (rank {self.device_rank})")
        self.update_status(RunnerIdle())

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def update_status(self, status: RunnerStatus) -> None:
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status)
        )

    def send_task_status(self, task_id: TaskId, task_status: TaskStatus) -> None:
        self.event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=task_status)
        )

    def acknowledge_task(self, task: Task) -> None:
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def main(self) -> None:
        with self.task_receiver:
            for task in self.task_receiver:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)
                self.handle_first_task(task)
                if isinstance(self.current_status, RunnerShutdown):
                    break

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def handle_first_task(self, task: Task) -> None:  # noqa: C901 — mirrors MLX runner
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(
                self.current_status, (RunnerIdle, RunnerFailed)
            ):
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                self._connect_to_group()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())
                logger.info(f"rank {self.device_rank} connected")

            case LoadModel() if isinstance(
                self.current_status, (RunnerConnected, RunnerIdle)
            ):
                total_layers = self.shard.end_layer - self.shard.start_layer
                self.update_status(RunnerLoading(layers_loaded=0, total_layers=total_layers))
                self.acknowledge_task(task)

                if self.device_rank == 0:
                    self._load_model()
                else:
                    logger.info(
                        f"rank {self.device_rank}: model served via RPC — no local load"
                    )

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())
                logger.info(f"rank {self.device_rank} loaded")

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)
                if self.device_rank == 0 and self._llm is not None:
                    self._warmup()
                logger.info(
                    f"runner initialized in {time.time() - self.setup_start_time:.1f}s"
                )
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerReady())
                logger.info(f"rank {self.device_rank} ready")

            case TextGeneration() if isinstance(self.current_status, RunnerReady):
                return_code = self.handle_generation_tasks(starting_task=task)
                if return_code == ExitCode.Shutdown:
                    return

            case Shutdown():
                self._shutdown(task)
                return

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine "
                    f"in {self.current_status=}"
                )

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------

    def _connect_to_group(self) -> None:
        """Rank 1+: start the RPC server.  Rank 0: nothing to do yet."""
        if self.device_rank != 0:
            port = self.llama_instance.rpc_port
            logger.info(f"rank {self.device_rank}: starting RPC server on port {port}")
            self._rpc_server_proc = start_rpc_server(port)
            # Give the server a moment to start accepting connections
            time.sleep(1.0)
            logger.info(f"rank {self.device_rank}: RPC server running (pid={self._rpc_server_proc.pid})")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Rank 0: resolve the GGUF path and load via llama-cpp-python."""
        model_path = find_gguf_model_path(self.shard)
        if model_path is None:
            card = self.shard.model_card
            raise RuntimeError(
                f"GGUF model file not found for {card.model_id}. "
                f"Expected gguf_repo_id={card.gguf_repo_id!r}, "
                f"gguf_filename={card.gguf_filename!r} in {self.shard}. "
                "Download the model first."
            )
        self._model_path = model_path
        self._llm = load_llamacpp_model(
            model_path=model_path,
            instance=self.llama_instance,
            bound_runner_id=self.runner_id,
            shard=self.shard,
        )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        """Run a short dummy completion to warm up the GPU kernels."""
        assert self._llm is not None
        try:
            list(
                self._llm.create_completion(  # type: ignore[call-arg]
                    prompt="Hello",
                    max_tokens=1,
                    stream=True,
                )
            )
            logger.info("warmup complete")
        except Exception as exc:
            logger.warning(f"warmup failed (non-fatal): {exc}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def handle_generation_tasks(self, starting_task: TextGeneration) -> ExitCode:
        assert isinstance(self.current_status, RunnerReady)
        self.update_status(RunnerRunning())
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)
        self.active_tasks[starting_task.task_id] = starting_task

        while self.active_tasks:
            # Process one task at a time (llama-cpp is synchronous)
            task_id, task = next(iter(self.active_tasks.items()))
            self._run_generation(task)
            self.active_tasks.pop(task_id, None)
            self.send_task_status(task_id, TaskStatus.Complete)

            try:
                next_task = self.task_receiver.receive_nowait()
                if next_task.task_id in self.seen:
                    continue
                self.seen.add(next_task.task_id)
                match next_task:
                    case TextGeneration():
                        self.acknowledge_task(next_task)
                        self.active_tasks[next_task.task_id] = next_task
                    case Shutdown():
                        self._shutdown(next_task)
                        return ExitCode.Shutdown
                    case _:
                        raise ValueError(
                            f"Received {next_task.__class__.__name__} during generation"
                        )
            except WouldBlock:
                pass

        self.update_status(RunnerReady())
        logger.info("runner ready")
        return ExitCode.AllTasksComplete

    def _run_generation(self, task: TextGeneration) -> None:
        """Stream tokens for a single TextGeneration task."""
        if self.device_rank != 0 or self._llm is None:
            # Non-rank-0 workers are passive (RPC server handles their GPU work)
            return

        params = task.task_params
        command_id = task.command_id

        # Format messages into a single prompt string
        prompt = _format_messages(params)

        stop: list[str] | str | None = params.stop
        temperature = params.temperature if params.temperature is not None else 1.0
        top_p = params.top_p if params.top_p is not None else 0.95
        max_tokens = params.max_output_tokens or 2048

        try:
            stream = self._llm.create_completion(  # type: ignore[call-arg]
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=True,
                echo=False,
            )

            prompt_tokens: int | None = None

            for chunk in stream:  # type: ignore[union-attr]
                if task.task_id in self._cancelled_tasks:
                    break

                choice = chunk["choices"][0]  # type: ignore[index]
                delta_text: str = choice.get("text", "") or ""  # type: ignore[union-attr]
                finish_reason = choice.get("finish_reason")  # type: ignore[union-attr]

                # llama-cpp doesn't stream token IDs; use 0 as placeholder
                token_id = 0

                # Final chunk may carry usage
                usage_data = chunk.get("usage")  # type: ignore[union-attr]
                if usage_data is not None:
                    prompt_tokens = usage_data.get("prompt_tokens", prompt_tokens)  # type: ignore[union-attr]

                if delta_text or finish_reason is not None:
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=TokenChunk(
                                model=self.model_id,
                                text=delta_text,
                                token_id=token_id,
                                usage=None,
                                finish_reason=finish_reason,
                                stats=None,
                                logprob=None,
                                top_logprobs=None,
                                is_thinking=False,
                            ),
                        )
                    )

        except Exception as exc:
            logger.opt(exception=exc).error("Generation error")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=ErrorChunk(
                        error_message=str(exc),
                        model=self.model_id,
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self, task: Task) -> None:
        logger.info("llama-cpp runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)

        if self._rpc_server_proc is not None:
            try:
                self._rpc_server_proc.terminate()
                self._rpc_server_proc.wait(timeout=5)
            except Exception:
                with contextlib.suppress(Exception):
                    self._rpc_server_proc.kill()

        self._llm = None
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())


# ------------------------------------------------------------------
# Prompt formatting
# ------------------------------------------------------------------

def _format_messages(params: TextGenerationTaskParams) -> str:
    """Format a list of InputMessages into a plain-text prompt.

    Uses a simple ChatML-style template.  llama-cpp-python supports the
    proper chat-template API via ``create_chat_completion``, but we use
    the lower-level ``create_completion`` for broader model compatibility.
    A follow-up can switch to ``create_chat_completion`` for models that
    declare a chat template in their tokenizer config.
    """
    parts: list[str] = []
    if params.instructions:
        parts.append(f"<|im_start|>system\n{params.instructions}<|im_end|>")
    for msg in params.input:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
