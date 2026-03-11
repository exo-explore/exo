import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
from enum import Enum

import httpx
from anyio import WouldBlock

from exo.shared.types.chunks import ErrorChunk, TokenChunk
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
from exo.shared.types.worker.instances import BoundInstance
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
from exo.shared.constants import EXO_MODELS_DIR
from exo.worker.runner.bootstrap import logger


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Shutdown = "Shutdown"


def _check_vllm_available() -> None:
    try:
        import vllm

        logger.info(f"vllm available: {vllm.__version__}")
    except ImportError as e:
        raise RuntimeError(f"vLLM is not installed: {e}") from e
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(f"PyTorch is not installed: {e}") from e
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — vLLM requires a CUDA GPU")
    logger.info(
        f"vLLM pre-flight: torch {torch.__version__}, "
        f"CUDA {torch.version.cuda}, "
        f"GPU {torch.cuda.get_device_name(0)}"
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]  # type: ignore
        return port


class Runner:
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

        self.instance, self.runner_id, self.shard_metadata = (
            self.bound_instance.instance,
            self.bound_instance.bound_runner_id,
            self.bound_instance.bound_shard,
        )
        self.model_id = self.shard_metadata.model_card.model_id
        self.model_path = EXO_MODELS_DIR / self.model_id.normalize()

        self.vllm_process: subprocess.Popen[bytes] | None = None
        self.vllm_port: int = 0
        self.vllm_base_url: str = ""

        self.seen: set[TaskId] = set()
        self.active_tasks: dict[TaskId, TextGeneration] = {}

        logger.info("hello from the vllm runner")
        _check_vllm_available()
        self.setup_start_time = time.time()
        self.update_status(RunnerIdle())

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task_id: TaskId, task_status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=task_status)
        )

    def acknowledge_task(self, task: Task):
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self):
        with self.task_receiver:
            for task in self.task_receiver:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)
                self.handle_first_task(task)
                if isinstance(self.current_status, RunnerShutdown):
                    break

    def handle_first_task(self, task: Task):
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(
                self.current_status, (RunnerIdle, RunnerFailed)
            ):
                logger.info("vllm runner connecting (no-op)")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())

            case LoadModel() if isinstance(
                self.current_status, (RunnerConnected, RunnerIdle)
            ):
                logger.info("vllm runner loading model")
                self.update_status(RunnerLoading(layers_loaded=0, total_layers=1))
                self.acknowledge_task(task)
                self._start_vllm_server()
                self._wait_for_vllm_health()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                logger.info("vllm runner warming up")
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerReady())
                logger.info(
                    f"vllm runner ready in {time.time() - self.setup_start_time:.1f}s"
                )

            case TextGeneration() if isinstance(self.current_status, RunnerReady):
                return_code = self.handle_generation_tasks(starting_task=task)
                if return_code == ExitCode.Shutdown:
                    return

            case Shutdown():
                self.shutdown(task)
                return

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                )

    def _start_vllm_server(self):
        self.vllm_port = _find_free_port()
        self.vllm_base_url = f"http://127.0.0.1:{self.vllm_port}"

        env = os.environ.copy()
        env["VLLM_SERVER_DEV_MODE"] = "1"

        cmd = [
            sys.executable,
            "-m",
            "exo.vllm_entry",
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.vllm_port),
            "--enable-sleep-mode",
        ]
        logger.info(f"Starting vLLM server on :{self.vllm_port} for {self.model_id}")
        self.vllm_process = subprocess.Popen(cmd, env=env, start_new_session=True)

        def _kill_vllm():
            if self.vllm_process and self.vllm_process.poll() is None:
                os.killpg(self.vllm_process.pid, signal.SIGKILL)

        atexit.register(_kill_vllm)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # type: ignore
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))  # type: ignore

    def _wait_for_vllm_health(self):
        while True:
            if self.vllm_process and self.vllm_process.poll() is not None:
                raise RuntimeError(
                    f"vLLM process died with code {self.vllm_process.returncode}"
                )
            try:
                resp = httpx.get(f"{self.vllm_base_url}/health", timeout=5)
                if resp.status_code == 200:
                    logger.info("vLLM server is healthy")
                    return
            except httpx.ConnectError:
                pass
            time.sleep(1)

    def handle_generation_tasks(self, starting_task: TextGeneration):
        self.update_status(RunnerRunning())
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)
        self.active_tasks[starting_task.task_id] = starting_task

        self._stream_generation(starting_task)
        self.active_tasks.pop(starting_task.task_id, None)

        while True:
            try:
                task = self.task_receiver.receive_nowait()
                if task.task_id in self.seen:
                    continue
                self.seen.add(task.task_id)

                match task:
                    case TextGeneration():
                        self.acknowledge_task(task)
                        self.active_tasks[task.task_id] = task
                        self._stream_generation(task)
                        self.active_tasks.pop(task.task_id, None)
                    case Shutdown():
                        self.shutdown(task)
                        return ExitCode.Shutdown
                    case _:
                        raise ValueError(f"Unexpected task {task.__class__.__name__}")
            except WouldBlock:
                break

        self.update_status(RunnerReady())
        return ExitCode.AllTasksComplete

    def _stream_generation(self, task: TextGeneration):
        params = task.task_params
        messages = [{"role": m.role, "content": m.content} for m in params.input]

        payload: dict[str, object] = {
            "model": str(params.model),
            "messages": messages,
            "stream": True,
        }
        if params.max_output_tokens is not None:
            payload["max_tokens"] = params.max_output_tokens
        if params.temperature is not None:
            payload["temperature"] = params.temperature
        if params.top_p is not None:
            payload["top_p"] = params.top_p
        if params.stop is not None:
            payload["stop"] = params.stop
        if params.seed is not None:
            payload["seed"] = params.seed

        try:
            with httpx.stream(
                "POST",
                f"{self.vllm_base_url}/v1/chat/completions",
                json=payload,
                timeout=300,
            ) as response:
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    parsed: dict[str, object] = json.loads(data)  # type: ignore
                    choices_list: list[object] = list(parsed.get("choices") or [])  # type: ignore
                    choice: dict[str, object] = choices_list[0] if choices_list else {}  # type: ignore
                    delta: dict[str, object] = choice.get("delta") or {}  # type: ignore
                    text = str(delta.get("content", ""))
                    finish_reason_val = choice.get("finish_reason")
                    finish_reason = (
                        str(finish_reason_val)
                        if isinstance(finish_reason_val, str)
                        else None
                    )

                    if text or finish_reason:
                        mapped_reason = (
                            finish_reason
                            if finish_reason in ("stop", "length", "content_filter")
                            else None
                        )
                        self.event_sender.send(
                            ChunkGenerated(
                                command_id=task.command_id,
                                chunk=TokenChunk(
                                    model=self.model_id,
                                    text=text or "",
                                    token_id=0,
                                    usage=None,
                                    finish_reason=mapped_reason,
                                ),
                            )
                        )

            self.send_task_status(task.task_id, TaskStatus.Complete)
        except Exception as e:
            logger.opt(exception=e).error("vLLM generation failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        error_message=str(e),
                        model=self.model_id,
                    ),
                )
            )
            self.send_task_status(task.task_id, TaskStatus.Complete)

    def shutdown(self, task: Task):
        logger.info("vllm runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        if self.vllm_process:
            os.killpg(self.vllm_process.pid, signal.SIGTERM)
            try:
                self.vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.vllm_process.pid, signal.SIGKILL)
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())
