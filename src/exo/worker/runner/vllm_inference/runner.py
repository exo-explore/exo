import gc
import time
from enum import Enum

import torch
import vllm
from anyio import WouldBlock
from vllm.engine.arg_utils import EngineArgs
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine

from exo.shared.constants import EXO_MODELS_DIR
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
from exo.shared.types.text_generation import TextGenerationTaskParams
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
from exo.worker.runner.bootstrap import logger


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Shutdown = "Shutdown"


def _check_vllm_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — vLLM requires a CUDA GPU")
    logger.info(
        f"vLLM pre-flight: vllm {vllm.__version__}, "
        f"torch {torch.__version__}, "
        f"CUDA {torch.version.cuda}, "
        f"GPU {torch.cuda.get_device_name(0)}"
    )


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

        self.engine: LLMEngine | None = None

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
                self._load_model()
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

    def _load_model(self):
        from exo.vllm_patches.growable_cache import patch_vllm

        patch_vllm()

        engine_args = EngineArgs(
            model=str(self.model_path),
            served_model_name=str(self.model_id),
            gpu_memory_utilization=0.05,
            trust_remote_code=self.shard_metadata.model_card.trust_remote_code,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        logger.info(f"vLLM engine loaded for {self.model_id}")

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        assert self.engine is not None
        tokenizer = self.engine.get_tokenizer()
        result = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(result, str)
        return result

    def _make_sampling_params(self, params: TextGenerationTaskParams) -> SamplingParams:
        kwargs: dict[str, object] = {}
        if params.max_output_tokens is not None:
            kwargs["max_tokens"] = params.max_output_tokens
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.stop is not None:
            kwargs["stop"] = params.stop
        if params.seed is not None:
            kwargs["seed"] = params.seed
        return SamplingParams(**kwargs)

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
        assert self.engine is not None
        params = task.task_params
        messages = [{"role": m.role, "content": m.content} for m in params.input]

        try:
            prompt = self._format_prompt(messages)
            sampling_params = self._make_sampling_params(params)
            request_id = str(task.task_id)

            self.engine.add_request(request_id, prompt, sampling_params)

            prev_text = ""
            while self.engine.has_unfinished_requests():
                outputs: list[RequestOutput] = self.engine.step()
                for output in outputs:
                    if output.request_id != request_id:
                        continue
                    completion = output.outputs[0]
                    delta = completion.text[len(prev_text) :]
                    prev_text = completion.text
                    finish_reason = completion.finish_reason

                    if delta or finish_reason:
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
                                    text=delta or "",
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
        del self.engine
        self.engine = None
        gc.collect()
        torch.cuda.empty_cache()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())
