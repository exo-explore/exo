import math
import resource
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

import mlx.core as mx
from anyio import WouldBlock
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.gpt_oss import Model as GptOssModel

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.chunks import (
    ErrorChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.mlx import Model
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
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
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
from exo.utils.channels import MpReceiver, MpSender, NonBlockingGenerator, mp_channel
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.generate import warmup_inference
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    detect_thinking_prompt_suffix,
    initialize_mlx,
    load_mlx_items,
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import BatchGenerator
from exo.worker.runner.llm_inference.model_output_parsers import (
    parse_deepseek_v32,
    parse_gpt_oss,
    parse_thinking_models,
    parse_tool_calls,
)

from .tool_parsers import ToolParser, make_mlx_parser

ALL_TASKS_COMPLETE_CODE = 0
SHUTDOWN_CODE = 1


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

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

        self.instance, self.runner_id, self.shard_metadata = (
            self.bound_instance.instance,
            self.bound_instance.bound_runner_id,
            self.bound_instance.bound_shard,
        )
        self.model_id = self.shard_metadata.model_card.model_id
        self.device_rank = self.shard_metadata.device_rank

        logger.info("hello from the runner")
        if getattr(self.shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(self.shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        self.setup_start_time = time.time()
        self.cancelled_tasks = set[TaskId]()

        self.inference_model: Model | None = None
        self.tokenizer = None
        self.tool_parser: ToolParser | None = None
        self.group = None
        self.kv_prefix_cache: KVPrefixCache | None = None
        self.check_for_cancel_every: int | None = None
        self.batch_generator: BatchGenerator | None = None

        self.seen: set[TaskId] = set()
        self.active_tasks: dict[
            TaskId,
            tuple[
                TextGeneration,
                NonBlockingGenerator[GenerationResponse | ToolCallResponse],
            ],
        ] = {}

        logger.info("runner created")
        self.update_status(RunnerIdle())

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task: Task, status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=status)
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
                self.cancelled_tasks.discard(TaskId("CANCEL_CURRENT_TASK"))
                self.handle_first_task(task)
                if isinstance(self.current_status, RunnerShutdown):
                    break

    def handle_first_task(self, task: Task):
        self.send_task_status(task, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(
                self.current_status, (RunnerIdle, RunnerFailed)
            ):
                logger.info("runner connecting")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)

                self.group = initialize_mlx(self.bound_instance)

                self.send_task_status(task, TaskStatus.Complete)
                self.update_status(RunnerConnected())
                logger.info("runner connected")

            # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
            case LoadModel():
                if (
                    isinstance(self.current_status, RunnerConnected)
                    and self.group is not None
                ) or (
                    isinstance(self.current_status, RunnerIdle) and self.group is None
                ):
                    total_layers = (
                        self.shard_metadata.end_layer - self.shard_metadata.start_layer
                    )
                    logger.info("runner loading")

                    self.update_status(
                        RunnerLoading(layers_loaded=0, total_layers=total_layers)
                    )
                    self.acknowledge_task(task)

                    def on_model_load_timeout() -> None:
                        self.update_status(
                            RunnerFailed(error_message="Model loading timed out")
                        )
                        time.sleep(0.5)

                    def on_layer_loaded(layers_loaded: int, total: int) -> None:
                        self.update_status(
                            RunnerLoading(
                                layers_loaded=layers_loaded, total_layers=total
                            )
                        )

                    assert (
                        ModelTask.TextGeneration in self.shard_metadata.model_card.tasks
                    ), (
                        f"Incorrect model task(s): {self.shard_metadata.model_card.tasks}"
                    )
                    self.inference_model, self.tokenizer = load_mlx_items(
                        self.bound_instance,
                        self.group,
                        on_timeout=on_model_load_timeout,
                        on_layer_loaded=on_layer_loaded,
                    )
                    logger.info(
                        f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
                    )
                    if self.tokenizer.has_tool_calling:
                        assert self.tokenizer.tool_call_start
                        assert self.tokenizer.tool_call_end
                        assert self.tokenizer.tool_parser  # pyright: ignore[reportAny]
                        self.tool_parser = make_mlx_parser(
                            self.tokenizer.tool_call_start,
                            self.tokenizer.tool_call_end,
                            self.tokenizer.tool_parser,  # pyright: ignore[reportAny]
                        )
                    self.kv_prefix_cache = KVPrefixCache(self.group)

                    self.send_task_status(task, TaskStatus.Complete)
                    self.update_status(RunnerLoaded())
                    logger.info("runner loaded")

            case StartWarmup():
                if isinstance(self.current_status, RunnerLoaded):
                    logger.info("runner warming up")

                    self.update_status(RunnerWarmingUp())
                    self.acknowledge_task(task)

                    logger.info(f"warming up inference for instance: {self.instance}")
                    assert self.inference_model
                    assert self.tokenizer

                    t = time.monotonic()
                    toks = warmup_inference(
                        model=self.inference_model,
                        tokenizer=self.tokenizer,
                        group=self.group,
                    )
                    logger.info(f"warmed up by generating {toks} tokens")
                    self.check_for_cancel_every = min(
                        math.ceil(toks / min(time.monotonic() - t, 0.001)), 100
                    )
                    if self.group is not None:
                        self.check_for_cancel_every = int(
                            mx.max(
                                mx.distributed.all_gather(
                                    mx.array([self.check_for_cancel_every]),
                                    group=self.group,
                                )
                            ).item()
                        )

                    logger.info(
                        f"runner checking for cancellation every {self.check_for_cancel_every} tokens"
                    )
                    logger.info(
                        f"runner initialized in {time.time() - self.setup_start_time} seconds"
                    )

                    self.batch_generator = BatchGenerator(
                        model=self.inference_model,
                        tokenizer=self.tokenizer,
                        group=self.group,
                        kv_prefix_cache=self.kv_prefix_cache,
                        model_id=self.model_id,
                        device_rank=self.device_rank,
                        cancel_receiver=self.cancel_receiver,
                        cancelled_tasks=self.cancelled_tasks,
                        event_sender=self.event_sender,
                        check_for_cancel_every=self.check_for_cancel_every,
                    )

                    self.send_task_status(task, TaskStatus.Complete)
                    self.update_status(RunnerReady())
                    logger.info("runner ready")

            case TextGeneration():
                return_code = self.handle_generation_tasks(starting_task=task)
                if return_code == SHUTDOWN_CODE:
                    return

            case Shutdown():
                self.shutdown(task)
                return

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                )

    def shutdown(self, task: Task):
        logger.info("runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        if not TYPE_CHECKING:
            del self.inference_model, self.tokenizer, self.group
            mx.clear_cache()
            import gc

            gc.collect()
        self.send_task_status(task, TaskStatus.Complete)
        self.update_status(RunnerShutdown())

    def get_non_blocking_generator(
        self,
        task: TextGeneration,
        receiver: MpReceiver[GenerationResponse],
    ) -> NonBlockingGenerator[GenerationResponse | ToolCallResponse]:
        assert self.tokenizer
        assert self.inference_model

        mlx_generator: Generator[GenerationResponse | ToolCallResponse | None] = (
            NonBlockingGenerator(receiver)
        )

        if self.tokenizer.has_thinking:
            prompt = apply_chat_template(self.tokenizer, task.task_params)
            mlx_generator = parse_thinking_models(
                mlx_generator,
                self.tokenizer,
                starts_in_thinking=detect_thinking_prompt_suffix(
                    prompt, self.tokenizer
                ),
            )

        if isinstance(self.inference_model, GptOssModel):
            mlx_generator = parse_gpt_oss(mlx_generator)
        elif (
            isinstance(self.inference_model, DeepseekV32Model)
            and "deepseek" in self.model_id.normalize().lower()
        ):
            mlx_generator = parse_deepseek_v32(mlx_generator)
        elif self.tool_parser:
            mlx_generator = parse_tool_calls(mlx_generator, self.tool_parser)

        return NonBlockingGenerator(mlx_generator)

    def submit_text_generation(self, task: TextGeneration):
        assert self.batch_generator is not None
        sender, receiver = mp_channel[GenerationResponse]()
        self.active_tasks[task.task_id] = (
            task,
            self.get_non_blocking_generator(task, receiver),
        )
        self.batch_generator.submit(task, sender)

    def handle_generation_tasks(self, starting_task: TextGeneration):
        assert isinstance(self.current_status, RunnerReady)
        assert self.batch_generator

        logger.info(f"received chat request: {starting_task}")
        self.update_status(RunnerRunning())
        logger.info("runner running")
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)

        self.submit_text_generation(starting_task)

        while self.active_tasks:
            self.batch_generator.step()

            finished: list[TaskId] = []
            for task_id, (text_task, gen) in self.active_tasks.items():
                response = gen.try_receive()
                if response is None:
                    if gen.is_exhausted:
                        finished.append(task_id)
                    continue

                self.send_response(response, text_task.command_id)

                was_cancelled = (text_task.task_id in self.cancelled_tasks) or (
                    TaskId("CANCEL_CURRENT_TASK") in self.cancelled_tasks
                )

                is_finished = (
                    isinstance(response, ToolCallResponse)
                    or response.finish_reason is not None
                )
                if is_finished or was_cancelled:
                    if not was_cancelled:
                        self.send_task_status(text_task, TaskStatus.Complete)
                    finished.append(task_id)

            for task_id in finished:
                del self.active_tasks[task_id]

            try:
                task = self.task_receiver.receive_nowait()

                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)
                self.cancelled_tasks.discard(TaskId("CANCEL_CURRENT_TASK"))

                match task:
                    case TextGeneration():
                        self.acknowledge_task(task)
                        self.submit_text_generation(task)
                    case Shutdown():
                        self.shutdown(task)
                        return SHUTDOWN_CODE
                    case _:
                        raise ValueError(
                            f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                        )

            except WouldBlock:
                pass

        self.update_status(RunnerReady())
        logger.info("runner ready")

        return ALL_TASKS_COMPLETE_CODE

    def send_response(
        self, response: GenerationResponse | ToolCallResponse, command_id: CommandId
    ):
        match response:
            case GenerationResponse():
                if self.device_rank == 0 and response.finish_reason == "error":
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=ErrorChunk(
                                error_message=response.text,
                                model=self.model_id,
                            ),
                        )
                    )

                elif self.device_rank == 0:
                    assert response.finish_reason not in (
                        "error",
                        "tool_calls",
                        "function_call",
                    )
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=TokenChunk(
                                model=self.model_id,
                                text=response.text,
                                token_id=response.token,
                                usage=response.usage,
                                finish_reason=response.finish_reason,
                                stats=response.stats,
                                logprob=response.logprob,
                                top_logprobs=response.top_logprobs,
                                is_thinking=response.is_thinking,
                            ),
                        )
                    )
            case ToolCallResponse():
                if self.device_rank == 0:
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=ToolCallChunk(
                                tool_calls=response.tool_calls,
                                model=self.model_id,
                                usage=response.usage,
                                stats=response.stats,
                            ),
                        )
                    )
