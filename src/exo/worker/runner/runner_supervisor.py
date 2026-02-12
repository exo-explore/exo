import contextlib
import os
import signal
import struct
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Process
from typing import Self

import anyio
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    to_thread,
)
from loguru import logger

from exo.shared.types.events import (
    Event,
    JacclSideChannelData,
    JacclSideChannelGathered,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.instances import BoundInstance, MlxJacclInstance
from exo.shared.types.worker.runners import (
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoading,
    RunnerRunning,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver, MpSender, Sender, mp_channel
from exo.worker.runner.bootstrap import entrypoint


def _pipe_read_exact(fd: int, n: int) -> bytes | None:
    """Read exactly n bytes from a file descriptor. Returns None on EOF."""
    data = b""
    while len(data) < n:
        chunk = os.read(fd, n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _pipe_write_all(fd: int, data: bytes) -> None:
    """Write all bytes to a file descriptor."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


PREFILL_TIMEOUT_SECONDS = 60
DECODE_TIMEOUT_SECONDS = 5


@dataclass(eq=False)
class RunnerSupervisor:
    shard_metadata: ShardMetadata
    bound_instance: BoundInstance
    runner_process: Process
    initialize_timeout: float
    _ev_recv: MpReceiver[Event]
    _task_sender: MpSender[Task]
    _event_sender: Sender[Event]
    _pipe_read_fd: int | None = None  # Python reads runner's pipe output
    _pipe_write_fd: int | None = None  # Python writes gathered data to runner
    _child_pipe_fds: tuple[int, int] | None = None  # fds to close after fork
    status: RunnerStatus = field(default_factory=RunnerIdle, init=False)
    pending: dict[TaskId, anyio.Event] = field(default_factory=dict, init=False)
    completed: set[TaskId] = field(default_factory=set, init=False)
    _gathered_waiters: dict[
        int, tuple[anyio.Event, JacclSideChannelGathered | None]
    ] = field(default_factory=dict, init=False)

    @classmethod
    def create(
        cls,
        *,
        bound_instance: BoundInstance,
        event_sender: Sender[Event],
        initialize_timeout: float = 400,
    ) -> Self:
        ev_send, ev_recv = mp_channel[Event]()
        # A task is kind of a runner command
        task_sender, task_recv = mp_channel[Task]()

        # For MlxJaccl instances, create pipe pairs for SideChannel relay.
        # Pipe pair 1: C++ writes local data → Python reads it
        # Pipe pair 2: Python writes gathered data → C++ reads it
        pipe_read_fd: int | None = None
        pipe_write_fd: int | None = None
        child_pipe_fds: tuple[int, int] | None = None
        pipe_fds_for_child: tuple[int, int] | None = None

        if isinstance(bound_instance.instance, MlxJacclInstance):
            python_reads, mlx_writes = os.pipe()  # C++ → Python
            mlx_reads, python_writes = os.pipe()  # Python → C++
            pipe_read_fd = python_reads
            pipe_write_fd = python_writes
            child_pipe_fds = (mlx_reads, mlx_writes)
            pipe_fds_for_child = (mlx_reads, mlx_writes)

        runner_process = Process(
            target=entrypoint,
            args=(
                bound_instance,
                ev_send,
                task_recv,
                logger,
                pipe_fds_for_child,
            ),
            daemon=True,
        )

        shard_metadata = bound_instance.bound_shard

        self = cls(
            bound_instance=bound_instance,
            shard_metadata=shard_metadata,
            runner_process=runner_process,
            initialize_timeout=initialize_timeout,
            _ev_recv=ev_recv,
            _task_sender=task_sender,
            _event_sender=event_sender,
            _pipe_read_fd=pipe_read_fd,
            _pipe_write_fd=pipe_write_fd,
            _child_pipe_fds=child_pipe_fds,
        )

        return self

    async def run(self):
        self.runner_process.start()

        # Close the child-side pipe fds in the parent process (child inherited them via fork)
        if self._child_pipe_fds is not None:
            for fd in self._child_pipe_fds:
                os.close(fd)
            self._child_pipe_fds = None

        if self._pipe_read_fd is not None:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._pipe_relay)
                tg.start_soon(self._forward_events)
        else:
            await self._forward_events()

    def shutdown(self):
        logger.info("Runner supervisor shutting down")
        self._ev_recv.close()
        self._task_sender.close()
        self._event_sender.close()
        self._close_pipe_fds()
        self.runner_process.join(1)
        if not self.runner_process.is_alive():
            logger.info("Runner process succesfully terminated")
            return

        # This is overkill but it's not technically bad, just unnecessary.
        logger.warning("Runner process didn't shutdown succesfully, terminating")
        self.runner_process.terminate()
        self.runner_process.join(1)
        if not self.runner_process.is_alive():
            return

        logger.critical("Runner process didn't respond to SIGTERM, killing")
        self.runner_process.kill()

        self.runner_process.join(1)
        if not self.runner_process.is_alive():
            return

        logger.critical(
            "Runner process didn't respond to SIGKILL. System resources may have leaked"
        )

    async def start_task(self, task: Task):
        if task.task_id in self.pending:
            logger.warning(
                f"Skipping invalid task {task} as it has already been submitted"
            )
            return
        if task.task_id in self.completed:
            logger.warning(
                f"Skipping invalid task {task} as it has already been completed"
            )
            return
        logger.info(f"Starting task {task}")
        event = anyio.Event()
        self.pending[task.task_id] = event
        try:
            await self._task_sender.send_async(task)
        except ClosedResourceError:
            logger.warning(f"Task {task} dropped, runner closed communication.")
            return
        await event.wait()

    async def _forward_events(self):
        with self._ev_recv as events:
            try:
                async for event in events:
                    if isinstance(event, RunnerStatusUpdated):
                        self.status = event.runner_status
                    if isinstance(event, TaskAcknowledged):
                        self.pending.pop(event.task_id).set()
                        continue
                    if (
                        isinstance(event, TaskStatusUpdated)
                        and event.task_status == TaskStatus.Complete
                    ):
                        # If a task has just been completed, we should be working on it.
                        assert isinstance(
                            self.status,
                            (
                                RunnerRunning,
                                RunnerWarmingUp,
                                RunnerLoading,
                                RunnerConnecting,
                                RunnerShuttingDown,
                            ),
                        )
                        self.completed.add(event.task_id)
                    await self._event_sender.send(event)
            except (ClosedResourceError, BrokenResourceError) as e:
                await self._check_runner(e)
                for tid in self.pending:
                    self.pending[tid].set()

    def _close_pipe_fds(self) -> None:
        if self._pipe_read_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._pipe_read_fd)
            self._pipe_read_fd = None
        if self._pipe_write_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._pipe_write_fd)
            self._pipe_write_fd = None
        if self._child_pipe_fds is not None:
            for fd in self._child_pipe_fds:
                with contextlib.suppress(OSError):
                    os.close(fd)
            self._child_pipe_fds = None

    async def _pipe_relay(self) -> None:
        """Relay JACCL SideChannel all_gather rounds between runner pipes and exo events."""
        assert self._pipe_read_fd is not None
        assert self._pipe_write_fd is not None
        read_fd = self._pipe_read_fd
        write_fd = self._pipe_write_fd
        sequence = 0

        try:
            while True:
                # 1. Read local data from runner: [uint32 size][size bytes]
                header = await to_thread.run_sync(partial(_pipe_read_exact, read_fd, 4))
                if header is None:
                    logger.info("JACCL pipe relay: runner closed pipe (EOF)")
                    break
                data_size: int = struct.unpack("<I", header)[0]  # pyright: ignore[reportAny]
                local_data = await to_thread.run_sync(
                    partial(_pipe_read_exact, read_fd, data_size)
                )
                if local_data is None:
                    logger.warning("JACCL pipe relay: EOF reading data payload")
                    break

                logger.info(
                    f"JACCL pipe relay: read {data_size} bytes from runner, seq={sequence}"
                )

                # 2. Emit JacclSideChannelData event
                waiter = anyio.Event()
                self._gathered_waiters[sequence] = (waiter, None)
                await self._event_sender.send(
                    JacclSideChannelData(
                        instance_id=self.bound_instance.instance.instance_id,
                        runner_id=self.bound_instance.bound_runner_id,
                        sequence=sequence,
                        data=local_data,
                    )
                )

                # 3. Wait for gathered result
                await waiter.wait()
                _, gathered_event = self._gathered_waiters.pop(sequence)
                assert gathered_event is not None

                # 4. Order gathered data by runner rank and concatenate
                instance = self.bound_instance.instance
                assert isinstance(instance, MlxJacclInstance)
                runner_order = list(instance.shard_assignments.runner_to_shard.keys())
                ordered_data = b"".join(
                    gathered_event.gathered_data[rid] for rid in runner_order
                )

                # 5. Write gathered data to runner: [uint32 total_size][total_size bytes]
                total_size = len(ordered_data)
                response = struct.pack("<I", total_size) + ordered_data
                await to_thread.run_sync(partial(_pipe_write_all, write_fd, response))

                logger.info(
                    f"JACCL pipe relay: wrote {total_size} bytes to runner, seq={sequence}"
                )
                sequence += 1
        except OSError as e:
            logger.warning(f"JACCL pipe relay: OS error: {e}")
        except Exception as e:
            logger.opt(exception=e).error("JACCL pipe relay: unexpected error")

    def notify_gathered(self, event: JacclSideChannelGathered) -> None:
        """Called by the worker when a JacclSideChannelGathered event arrives."""
        seq = event.sequence
        if seq not in self._gathered_waiters:
            logger.warning(f"JACCL: received gathered event for unknown sequence {seq}")
            return
        waiter, _ = self._gathered_waiters[seq]
        self._gathered_waiters[seq] = (waiter, event)
        waiter.set()

    def __del__(self) -> None:
        if self.runner_process.is_alive():
            logger.warning("RunnerSupervisor was not stopped cleanly.")
            with contextlib.suppress(ValueError):
                self.runner_process.kill()

    async def _check_runner(self, e: Exception) -> None:
        logger.info("Checking runner's status")
        if self.runner_process.is_alive():
            logger.info("Runner was found to be alive, attempting to join process")
            await to_thread.run_sync(self.runner_process.join, 1)
        rc = self.runner_process.exitcode
        logger.info(f"RunnerSupervisor exited with exit code {rc}")
        if rc == 0:
            return

        if isinstance(rc, int) and rc < 0:
            sig = -rc
            try:
                cause = f"signal={sig} ({signal.strsignal(sig)})"
            except Exception:
                cause = f"signal={sig}"
        else:
            cause = f"exitcode={rc}"

        logger.opt(exception=e).error(f"Runner terminated ({cause})")

        await self._event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=f"Terminated ({cause})"),
            )
        )
        self.shutdown()
