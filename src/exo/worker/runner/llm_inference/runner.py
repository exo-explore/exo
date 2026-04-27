import os
import queue
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum

import mlx.core as mx
from anyio import ClosedResourceError, EndOfStream
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.chunks import GenerationChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.mlx import KVCacheType, Model
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
    ModelLoadingResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
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
from exo.utils.ports import random_ephemeral_port
from exo.worker.engines.mlx.cache import KVPrefixCache, make_kv_cache
from exo.worker.engines.mlx.disaggregated.adapter import (
    serialize_mlx_cache_to_payload,
)
from exo.worker.engines.mlx.disaggregated.server import (
    PrefillJob,
    PrefillServer,
)
from exo.worker.engines.mlx.generator.generate import prefill as mlx_prefill
from exo.worker.engines.mlx.utils_mlx import (
    fix_unmatched_think_end_tokens,
    initialize_mlx,
    load_mlx_items,
)
from exo.worker.engines.mlx.vision import VisionProcessor
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    InferenceGenerator,
    SequentialGenerator,
)

from .batch_generator import Cancelled, Finished
from .tool_parsers import make_mlx_parser

PREFILL_PICKUP_TIMEOUT_SECONDS = 3
PREFILL_FINISH_TIMEOUT_SECONDS = 300


def _wire_dtype_from_cache(cache: KVCacheType) -> str:
    for c in cache:
        keys: mx.array | None = getattr(c, "keys", None)
        if keys is None:
            continue
        if keys.dtype == mx.bfloat16:
            return "bfloat16"
        if keys.dtype == mx.float16:
            return "float16"
        if keys.dtype == mx.float32:
            return "float32"
        break
    return "bfloat16"


@dataclass
class _PrefillRequest:
    job: PrefillJob
    started: threading.Event
    done: threading.Event
    holder: list[bytes | None]


_TaskStreamClosed = object()
WorkItem = Task | _PrefillRequest | object


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Shutdown = "Shutdown"


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
        self.device_rank = self.shard_metadata.device_rank

        logger.info("hello from the runner")
        if getattr(self.shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(self.shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        self.setup_start_time = time.time()

        self.generator: Builder | InferenceGenerator = Builder(
            self.model_id,
            self.event_sender,
            self.cancel_receiver,
        )

        self.seen: set[TaskId] = set()
        self.active_tasks: dict[
            TaskId,
            TextGeneration,
        ] = {}

        self._prefill_server: PrefillServer | None = None
        self._prefill_server_port: int | None = None
        self._work_queue: queue.Queue[WorkItem] = queue.Queue()
        self._task_reader_thread: threading.Thread | None = None

        logger.info("runner created")
        self.update_status(RunnerIdle())

    def _start_prefill_server(self) -> int:
        if self._prefill_server_port is not None:
            return self._prefill_server_port

        def resolve(job: PrefillJob) -> bytes | None:
            req = _PrefillRequest(
                job=job,
                started=threading.Event(),
                done=threading.Event(),
                holder=[None],
            )
            self._work_queue.put(req)
            if not req.started.wait(timeout=PREFILL_PICKUP_TIMEOUT_SECONDS):
                logger.warning(
                    f"Prefill request {job.request_id} not picked up within "
                    f"{PREFILL_PICKUP_TIMEOUT_SECONDS}s — runner busy"
                )
                return None
            if not req.done.wait(timeout=PREFILL_FINISH_TIMEOUT_SECONDS):
                logger.warning(
                    f"Prefill request {job.request_id} did not finish within "
                    f"{PREFILL_FINISH_TIMEOUT_SECONDS}s"
                )
                return None
            return req.holder[0]

        self._prefill_server = PrefillServer(
            resolve=resolve, host="0.0.0.0", port=random_ephemeral_port()
        )
        self._prefill_server.start()
        self._prefill_server_port = self._prefill_server.port
        return self._prefill_server_port

    def _start_task_reader(self) -> None:
        if self._task_reader_thread is not None:
            return

        def loop() -> None:
            try:
                with self.task_receiver:
                    for task in self.task_receiver:
                        self._work_queue.put(task)
            except (EndOfStream, ClosedResourceError):
                pass
            finally:
                self._work_queue.put(_TaskStreamClosed)

        self._task_reader_thread = threading.Thread(
            target=loop, name="task-reader", daemon=True
        )
        self._task_reader_thread.start()

    def _serve_prefill(self, req: _PrefillRequest) -> None:
        req.started.set()
        was_ready = isinstance(self.current_status, RunnerReady)
        if was_ready:
            self.update_status(RunnerRunning())
        try:
            req.holder[0] = self._serve_prefill_request(req.job)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to serve prefill request {req.job.request_id}"
            )
            req.holder[0] = None
        finally:
            req.done.set()
            if was_ready:
                self.update_status(
                    RunnerReady(prefill_server_port=self._prefill_server_port)
                )

    def _serve_prefill_request(self, job: PrefillJob) -> bytes:
        assert isinstance(self.generator, InferenceGenerator)
        model = self.generator.model
        tokenizer = self.generator.tokenizer
        group = self.generator.group

        prompt_tokens = mx.array(job.token_ids)
        prompt_tokens = fix_unmatched_think_end_tokens(prompt_tokens, tokenizer)
        n_tokens = int(prompt_tokens.shape[0])
        logger.info(
            f"Serving prefill: request_id={job.request_id} tokens={n_tokens} "
            f"start_pos={job.start_pos}"
        )
        t0 = time.perf_counter()

        cache = make_kv_cache(model)
        sampler = make_sampler(temp=1.0)
        prefill_input = prompt_tokens[:-2] if n_tokens > 2 else prompt_tokens
        _ = mlx_prefill(
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
            prompt_tokens=prefill_input,
            cache=cache,
            group=group,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        payload = serialize_mlx_cache_to_payload(
            cache,
            dtype=_wire_dtype_from_cache(cache),
            model_id=job.model_id,
            request_id=job.request_id,
            start_pos=job.start_pos,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Served prefill: request_id={job.request_id} "
            f"{n_tokens} tokens in {elapsed * 1000:.0f}ms "
            f"({n_tokens / max(elapsed, 0.001):.0f} tok/s)"
        )
        return payload

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
        self._start_task_reader()
        while True:
            item = self._work_queue.get()
            if item is _TaskStreamClosed:
                break
            if isinstance(item, _PrefillRequest):
                self._serve_prefill(item)
                continue
            task: Task = item  # type: ignore[assignment]
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
            case ConnectToGroup() if isinstance(self.current_status, RunnerIdle):
                assert isinstance(self.generator, Builder)
                logger.info("runner connecting")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)

                self.generator.group = initialize_mlx(self.bound_instance)

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())
                logger.info("runner connected")

            # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
            case LoadModel() if isinstance(self.generator, Builder) and (
                (
                    isinstance(self.current_status, RunnerConnected)
                    and self.generator.group is not None
                )
                or (
                    isinstance(self.current_status, RunnerIdle)
                    and self.generator.group is None
                )
            ):
                total_layers = (
                    self.shard_metadata.end_layer - self.shard_metadata.start_layer
                )
                logger.info("runner loading")

                self.update_status(
                    RunnerLoading(layers_loaded=0, total_layers=total_layers)
                )
                self.acknowledge_task(task)

                assert (
                    ModelTask.TextGeneration in self.shard_metadata.model_card.tasks
                ), f"Incorrect model task(s): {self.shard_metadata.model_card.tasks}"

                def load_model() -> Generator[ModelLoadingResponse]:
                    assert isinstance(self.generator, Builder)
                    (
                        self.generator.inference_model,
                        self.generator.tokenizer,
                        self.generator.vision_processor,
                    ) = yield from load_mlx_items(
                        self.bound_instance,
                        self.generator.group,
                    )

                for load_resp in load_model():
                    self.update_status(
                        RunnerLoading(
                            layers_loaded=load_resp.layers_loaded,
                            total_layers=load_resp.total,
                        )
                    )

                self.generator = self.generator.build()

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())
                logger.info("runner loaded")

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                assert isinstance(self.generator, InferenceGenerator)
                logger.info("runner warming up")

                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)

                self.generator.warmup()

                logger.info(
                    f"runner initialized in {time.time() - self.setup_start_time} seconds"
                )

                prefill_port = self._start_prefill_server()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerReady(prefill_server_port=prefill_port))
                logger.info("runner ready")

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

    def shutdown(self, task: Task):
        logger.info("runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        if isinstance(self.generator, InferenceGenerator):
            self.generator.close()
        mx.clear_cache()
        import gc

        gc.collect()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())

    def submit_text_generation(self, task: TextGeneration):
        assert isinstance(self.generator, InferenceGenerator)
        self.active_tasks[task.task_id] = task
        self.generator.submit(task)

    def handle_generation_tasks(self, starting_task: TextGeneration):
        assert isinstance(self.current_status, RunnerReady)
        assert isinstance(self.generator, InferenceGenerator)

        logger.info(f"received chat request: {starting_task}")
        self.update_status(RunnerRunning())
        logger.info("runner running")
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)

        self.submit_text_generation(starting_task)

        while self.active_tasks:
            results = self.generator.step()

            finished: list[TaskId] = []
            for task_id, result in results:
                match result:
                    case Cancelled():
                        finished.append(task_id)
                    case Finished():
                        self.send_task_status(task_id, TaskStatus.Complete)
                        finished.append(task_id)
                    case _:
                        self.send_chunk(result, self.active_tasks[task_id].command_id)

            for task_id in finished:
                self.active_tasks.pop(task_id, None)

            try:
                item = self._work_queue.get_nowait()
            except queue.Empty:
                continue
            if item is _TaskStreamClosed:
                # Task stream closed mid-generation. Bail out.
                return ExitCode.Shutdown
            if isinstance(item, _PrefillRequest):
                # Refuse — runner is mid-generation. Client picks up the 3s
                # pickup timeout and 503s.
                item.started.set()
                item.holder[0] = None
                item.done.set()
                continue
            task: Task = item  # type: ignore[assignment]
            if task.task_id in self.seen:
                logger.warning("repeat task - potential error")
                continue
            self.seen.add(task.task_id)
            match task:
                case TextGeneration():
                    self.acknowledge_task(task)
                    self.submit_text_generation(task)
                case Shutdown():
                    self.shutdown(task)
                    return ExitCode.Shutdown
                case _:
                    raise ValueError(
                        f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                    )

        self.update_status(RunnerReady(prefill_server_port=self._prefill_server_port))
        logger.info("runner ready")

        return ExitCode.AllTasksComplete

    def send_chunk(
        self,
        chunk: GenerationChunk,
        command_id: CommandId,
    ):
        if self.device_rank == 0:
            self.event_sender.send(ChunkGenerated(command_id=command_id, chunk=chunk))


@dataclass
class Builder:
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None
    vision_processor: VisionProcessor | None = None

    def build(
        self,
    ) -> InferenceGenerator:
        assert self.model_id
        assert self.inference_model
        assert self.tokenizer

        vision_processor = self.vision_processor

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        kv_prefix_cache = KVPrefixCache(self.group)

        device_rank = 0 if self.group is None else self.group.rank()
        if os.environ.get("EXO_NO_BATCH"):
            logger.info("using SequentialGenerator (batching disabled)")
            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )
        logger.info("using BatchGenerator")
        return BatchGenerator(
            model=self.inference_model,
            tokenizer=self.tokenizer,
            group=self.group,
            tool_parser=tool_parser,
            kv_prefix_cache=kv_prefix_cache,
            model_id=self.model_id,
            device_rank=device_rank,
            cancel_receiver=self.cancel_receiver,
            event_sender=self.event_sender,
            vision_processor=vision_processor,
        )
