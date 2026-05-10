import os
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO

from anyio import ClosedResourceError, EndOfStream

from exo.shared.constants import ENABLE_DISAGGREGATION
from exo.shared.types.chunks import Chunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ConnectToGroup,
    GenerationTask,
    ImageEdits,
    ImageGeneration,
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
    CancelledResponse,
    FinishedResponse,
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
from exo.worker.disaggregated.server import (
    PrefillRequest,
    PrefillServer,
)
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger

PREFILL_PICKUP_TIMEOUT_SECONDS = 3

# Window the runner blocks on ``_work_queue`` after the initial task
# is admitted, looking for sibling burst-arrivals that should land in
# the same ``SequentialGenerator._admit_queued_tasks`` window so their
# prefills can be batched.
#
# Empirically (3-node TB-RDMA Big Brain, gemma-4-26b-a4b-it-4bit on
# smbpt, 2 concurrent client requests dispatched within microseconds
# at the bench harness): the master process records both
# ``Executing command: TextGeneration`` events 15-33ms apart, but
# they reach the runner subprocess's ``_work_queue`` 150-200ms apart
# because of libp2p pubsub fan-out + mp-channel hop from the worker
# process to the runner subprocess. The original 20ms default
# missed slot #2 by ~130ms and ``batched_prefill`` never fired.
# 200ms catches it reliably; the cost is +200ms TTFT for genuinely
# solo requests, but the burst-coalesce only runs ONCE per
# ``handle_generation_tasks`` entry (i.e. only when transitioning
# from RunnerReady -> RunnerRunning, not on every admit), so
# back-to-back requests on a warm instance pay this only on the
# first wave. Set ``EXO_BURST_COALESCE_MS=0`` to disable
# (per-slot prefill on every request).
EXO_BURST_COALESCE_MS = "EXO_BURST_COALESCE_MS"
DEFAULT_BURST_COALESCE_MS = 200


def _parse_burst_coalesce_ms() -> int:
    raw = os.environ.get(EXO_BURST_COALESCE_MS)
    if raw is None:
        return DEFAULT_BURST_COALESCE_MS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            f"{EXO_BURST_COALESCE_MS}={raw!r} is not a valid int; "
            f"falling back to {DEFAULT_BURST_COALESCE_MS}ms"
        )
        return DEFAULT_BURST_COALESCE_MS
    return max(0, value)


PREFILL_FINISH_TIMEOUT_SECONDS = 300


@dataclass
class PrefillTask:
    request: PrefillRequest
    wfile: BinaryIO
    started: threading.Event
    done: threading.Event


class _TaskStreamClosed:
    pass


WorkItem = Task | PrefillTask | _TaskStreamClosed


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Shutdown = "Shutdown"


class Runner:
    def __init__(
        self,
        bound_instance: BoundInstance,
        builder: Builder,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
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

        self.generator: Builder | Engine = builder

        self.seen: set[TaskId] = set()
        self.active_tasks: dict[
            TaskId,
            GenerationTask,
        ] = {}

        self._prefill_server: PrefillServer | None = None
        self._prefill_server_port: int | None = None
        self._work_queue: queue.Queue[WorkItem] = queue.Queue()
        # Slot for a non-generation item picked up by
        # ``_coalesce_burst_generation_tasks`` -- consumed by the main
        # loop in ``handle_generation_tasks`` before its next
        # ``_work_queue.get_nowait()`` so the FIFO order between burst
        # text-gens and a trailing ``Shutdown`` / ``PrefillTask`` /
        # ``_TaskStreamClosed`` is preserved.
        self._burst_deferred_item: WorkItem | None = None
        self._task_reader_thread: threading.Thread | None = None

        logger.info("runner created")
        self.update_status(RunnerIdle())

    def _start_prefill_server(self) -> int | None:
        if not ENABLE_DISAGGREGATION:
            return None
        if self.device_rank != 0:
            return None
        if self._prefill_server_port is not None:
            return self._prefill_server_port

        def resolve(request: PrefillRequest, wfile: BinaryIO) -> bool:
            req = PrefillTask(
                request=request,
                wfile=wfile,
                started=threading.Event(),
                done=threading.Event(),
            )
            self._work_queue.put(req)
            if not req.started.wait(timeout=PREFILL_PICKUP_TIMEOUT_SECONDS):
                logger.warning(
                    f"Prefill request {request.request_id} not picked up within "
                    f"{PREFILL_PICKUP_TIMEOUT_SECONDS}s — runner busy"
                )
                return False
            if not req.done.wait(timeout=PREFILL_FINISH_TIMEOUT_SECONDS):
                logger.warning(
                    f"Prefill request {request.request_id} did not finish within "
                    f"{PREFILL_FINISH_TIMEOUT_SECONDS}s"
                )
            return True

        port = random_ephemeral_port()
        self._prefill_server = PrefillServer(resolve=resolve, host="0.0.0.0", port=port)
        self._prefill_server_port = port
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
                self._work_queue.put(_TaskStreamClosed())

        self._task_reader_thread = threading.Thread(target=loop, name="task-reader")
        self._task_reader_thread.start()

    def _serve_prefill(self, req: PrefillTask) -> None:
        req.started.set()
        try:
            assert isinstance(self.generator, Engine)
            self.generator.serve_prefill(req.request, req.wfile)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to serve prefill request {req.request.request_id}"
            )
        finally:
            req.done.set()

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
        try:
            while True:
                item = self._work_queue.get()
                if isinstance(item, _TaskStreamClosed):
                    break
                if isinstance(item, PrefillTask):
                    self._serve_prefill(item)
                    continue
                if item.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(item.task_id)
                self.handle_first_task(item)
                if isinstance(self.current_status, RunnerShutdown):
                    break
        finally:
            if self._prefill_server is not None:
                self._prefill_server.stop()
                self._prefill_server = None
            self.task_receiver.close()
            if self._task_reader_thread is not None:
                self._task_reader_thread.join(timeout=5)
                self._task_reader_thread = None

    def handle_first_task(self, task: Task):
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(self.current_status, RunnerIdle):
                assert isinstance(self.generator, Builder)
                logger.info("runner connecting")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)

                self.generator.connect(self.bound_instance)

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())
                logger.info("runner connected")

            # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
            case LoadModel() if isinstance(self.generator, Builder) and (
                isinstance(self.current_status, (RunnerConnected, RunnerIdle))
            ):
                total_layers = (
                    self.shard_metadata.end_layer - self.shard_metadata.start_layer
                )
                logger.info("runner loading")

                self.update_status(
                    RunnerLoading(layers_loaded=0, total_layers=total_layers)
                )
                self.acknowledge_task(task)

                for load_progress in self.generator.load(self.bound_instance):
                    self.update_status(
                        RunnerLoading(
                            layers_loaded=load_progress.layers_loaded,
                            total_layers=load_progress.total,
                        )
                    )

                self.generator = self.generator.build()

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())
                logger.info("runner loaded")

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                assert isinstance(self.generator, Engine)
                logger.info("runner warming up")

                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)

                self.generator.warmup()

                logger.info(
                    f"runner initialized in {time.time() - self.setup_start_time} seconds"
                )

                self._start_prefill_server()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(
                    RunnerReady(prefill_server_port=self._prefill_server_port)
                )
                logger.info("runner ready")

            case TextGeneration() | ImageEdits() | ImageGeneration() if isinstance(
                self.current_status, RunnerReady
            ):
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
        self.generator.close()
        import gc

        gc.collect()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())

    def submit_generation(self, task: GenerationTask):
        assert isinstance(self.generator, Engine)
        self.active_tasks[task.task_id] = task
        self.generator.submit(task)

    def _drain_pending_work_items(self, max_drain: int = 32) -> "ExitCode | None":
        """Non-blocking drain of immediately-available ``_work_queue`` items.

        Called between every ``step()`` iteration in the main generation
        loop. Submits ``GenerationTask`` siblings via the existing
        ``submit_generation`` path so the next ``step()``'s
        ``agree_on_tasks`` + ``_admit_queued_tasks`` sees them all in
        the same admit window (this is what extends ``batched_prefill``
        coverage past the initial 2-slot burst -- e.g. concurrency=4
        where the 3rd and 4th slots straggle ~1s behind the first
        pair).

        Specials end the drain and are handled in arrival order:

        * :class:`_TaskStreamClosed` -> return :attr:`ExitCode.Shutdown`
          to break the main loop.
        * :class:`PrefillTask` -> serve it (synchronous, blocks until
          done) then return ``None`` so the main loop continues.
        * :class:`Shutdown` -> shut the runner down and return
          :attr:`ExitCode.Shutdown`.

        Returns ``None`` to signal "keep looping" (queue exhausted or
        only generation tasks were drained), an ``ExitCode`` to signal
        the main loop should exit.

        ``max_drain`` is a defensive bound. In practice the queue
        carries 1-4 burst tasks at a time; the drain returns far
        sooner via ``queue.Empty``.
        """
        for _ in range(max_drain):
            if self._burst_deferred_item is not None:
                item = self._burst_deferred_item
                self._burst_deferred_item = None
            else:
                try:
                    item = self._work_queue.get_nowait()
                except queue.Empty:
                    return None
            if isinstance(item, _TaskStreamClosed):
                return ExitCode.Shutdown
            if isinstance(item, PrefillTask):
                self._serve_prefill(item)
                # ``_serve_prefill`` is synchronous; we yield back to
                # the main loop here so the next ``step()`` runs
                # before we drain more items, matching the
                # pre-refactor cadence where one ``PrefillTask`` per
                # iteration was the maximum.
                return None
            if item.task_id in self.seen:
                logger.warning("repeat task - potential error")
                continue
            self.seen.add(item.task_id)
            match item:
                case TextGeneration() | ImageGeneration() | ImageEdits():
                    self.acknowledge_task(item)
                    self.submit_generation(item)
                case Shutdown():
                    self.shutdown(item)
                    return ExitCode.Shutdown
                case _:
                    raise ValueError(
                        f"Received {item.__class__.__name__} outside of "
                        f"state machine in {self.current_status=}"
                    )
        return None

    def _coalesce_burst_generation_tasks(self, max_drain: int = 32) -> None:
        """Pull pending ``GenerationTask`` items into the generator's queue.

        Called from :meth:`handle_generation_tasks` after the initial
        ``submit_generation`` so the upcoming ``step()`` call admits the
        full burst together. Stops at the first non-generation item
        (``PrefillTask`` / ``_TaskStreamClosed`` / ``Shutdown``) and
        stashes that item in :attr:`_burst_deferred_item` so the main
        loop sees it before its next ``_work_queue.get_nowait()`` --
        re-queueing at the tail would race with the listener thread
        and silently re-order ``Shutdown`` past burst tasks.

        After draining whatever is immediately available, blocks on the
        queue for up to ``EXO_BURST_COALESCE_MS`` (default 20ms) to
        catch sibling burst-arrivals whose libp2p delivery straggles
        behind the first request -- without this, two concurrent
        client requests reliably miss the same admit window because
        only the first arrives before the runner reaches ``step()``.

        ``max_drain`` is a defensive bound so a saturated upstream
        producer can't starve the first ``step()`` indefinitely; in
        practice the work queue carries 1-2 burst-tasks at a time.
        """
        budget_ms = _parse_burst_coalesce_ms()
        deadline = time.monotonic() + budget_ms / 1000.0 if budget_ms > 0 else None
        drained = 0
        start = time.monotonic()
        for _ in range(max_drain):
            try:
                item = self._work_queue.get_nowait()
            except queue.Empty:
                if deadline is None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._work_queue.get(timeout=remaining)
                except queue.Empty:
                    break
            if isinstance(item, TextGeneration | ImageGeneration | ImageEdits):
                if item.task_id in self.seen:
                    continue
                self.seen.add(item.task_id)
                self.acknowledge_task(item)
                self.submit_generation(item)
                drained += 1
                continue
            self._burst_deferred_item = item
            break
        elapsed_ms = (time.monotonic() - start) * 1000.0
        # ``info`` when we actually batched (drained>=1) so operators see the
        # value the coalesce delivered; ``debug`` when nothing batched, so
        # solo-request runners stay quiet.
        if drained >= 1:
            logger.info(
                f"burst-coalesce drained={drained} budget_ms={budget_ms} "
                f"elapsed_ms={elapsed_ms:.1f} "
                f"deferred={self._burst_deferred_item is not None}"
            )
        else:
            logger.debug(
                f"burst-coalesce drained=0 budget_ms={budget_ms} "
                f"elapsed_ms={elapsed_ms:.1f} "
                f"deferred={self._burst_deferred_item is not None}"
            )

    def handle_generation_tasks(self, starting_task: GenerationTask):
        assert isinstance(self.current_status, RunnerReady)
        assert isinstance(self.generator, Engine)

        # Log identifiers only. The full ``starting_task`` is a deep
        # Pydantic model whose default ``__str__`` recursively repr's
        # every field (including ``chat_template_messages`` and any
        # nested token / image structures). On a multi-rank target
        # placement the worker plans the same TextGeneration repeatedly
        # while a runner is busy, so logging the full model on every
        # entry has been observed to peg rank 0 inside ``list_repr`` /
        # ``long_to_decimal_string`` for minutes (peak physical
        # footprint ~300 GB) and prevent it from ever entering the
        # model forward -- which the peer rank then deadlocks on inside
        # the first TP collective.
        logger.info(
            "received chat request task_id="
            f"{starting_task.task_id} command_id={starting_task.command_id} "
            f"task_type={starting_task.__class__.__name__}"
        )
        self.update_status(RunnerRunning())
        logger.info("runner running")
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)

        self.submit_generation(starting_task)

        # Coalesce burst-arrivals: drain TextGeneration / ImageGeneration /
        # ImageEdits items already sitting in ``_work_queue`` and submit
        # them BEFORE the first ``step()``. Without this, two concurrent
        # client requests that arrive within a few ms see the runner
        # admit task #1 alone (its prefill starts on the very first
        # ``step()``) and task #2 only joins on the next iteration --
        # which defeats batched-prefill admission entirely (the
        # ``_admit_queued_tasks`` candidate list never has B>=2 tasks).
        # Non-task items (PrefillTask / _TaskStreamClosed / Shutdown)
        # are left in the queue so the main loop's match block handles
        # them in order; we stop draining at the first non-task item to
        # preserve queue ordering.
        self._coalesce_burst_generation_tasks()

        while self.active_tasks:
            results = self.generator.step()

            finished: list[TaskId] = []
            for task_id, result in results:
                match result:
                    case CancelledResponse():
                        finished.append(task_id)
                    case FinishedResponse():
                        self.send_task_status(task_id, TaskStatus.Complete)
                        finished.append(task_id)
                    case other:
                        self.send_chunk(other, self.active_tasks[task_id].command_id)

            for task_id in finished:
                self.active_tasks.pop(task_id, None)

            # Drain ALL immediately-available items so concurrent
            # burst-arrivals that landed during the previous
            # ``step()`` (e.g. slots 3/4 of a concurrency=4 wave that
            # arrived behind slots 1/2 by libp2p straggle) are
            # submitted before the NEXT ``step()`` runs
            # ``agree_on_tasks`` + ``_admit_queued_tasks``. Without
            # this, the original code drained one item per iteration,
            # so the second admit cycle still saw a single candidate
            # and fell through to per-slot prefill -- we lose
            # batched-prefill on every slot beyond the first wave.
            #
            # Specials (``_TaskStreamClosed`` / ``PrefillTask`` /
            # ``Shutdown``) terminate the drain and are handled in
            # arrival order. The ``_burst_deferred_item`` slot is
            # checked first for FIFO preservation against the entry-
            # time burst-coalesce.
            exit_code = self._drain_pending_work_items()
            if exit_code is not None:
                return exit_code

        self.update_status(RunnerReady(prefill_server_port=self._prefill_server_port))
        logger.info("runner ready")

        return ExitCode.AllTasksComplete

    def send_chunk(
        self,
        chunk: Chunk,
        command_id: CommandId,
    ):
        assert isinstance(self.generator, Engine)
        self.event_sender.send(ChunkGenerated(command_id=command_id, chunk=chunk))
