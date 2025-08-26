import asyncio
import time
from asyncio import Queue
from functools import partial
from typing import AsyncGenerator, Optional

from loguru import logger

from exo.shared.db.sqlite import AsyncSQLiteEventStorage
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    InstanceDeleted,
    RunnerDeleted,
    RunnerStatusUpdated,
    TaskFailed,
    TaskStateUpdated,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus
from exo.shared.types.worker.common import RunnerId
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadPending,
    DownloadProgressData,
)
from exo.shared.types.worker.ops import (
    AssignRunnerOp,
    ExecuteTaskOp,
    RunnerDownOp,
    RunnerFailedOp,
    RunnerOp,
    RunnerOpType,
    RunnerUpOp,
    UnassignRunnerOp,
)
from exo.shared.types.worker.runners import (
    DownloadingRunnerStatus,
    FailedRunnerStatus,
    InactiveRunnerStatus,
    LoadedRunnerStatus,
    RunningRunnerStatus,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.common import AssignedRunner
from exo.worker.download.shard_downloader import RepoDownloadProgress, ShardDownloader
from exo.worker.runner.runner_supervisor import RunnerSupervisor


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        shard_downloader: ShardDownloader,
        worker_events: AsyncSQLiteEventStorage | None,
        global_events: AsyncSQLiteEventStorage | None,
    ):
        self.node_id: NodeId = node_id
        self.state: State = State()
        self.shard_downloader: ShardDownloader = shard_downloader
        self.worker_events: AsyncSQLiteEventStorage | None = (
            worker_events  # worker_events is None in some tests.
        )
        self.global_events: AsyncSQLiteEventStorage | None = global_events

        self.assigned_runners: dict[RunnerId, AssignedRunner] = {}
        self._task: asyncio.Task[None] | None = None

    ## Op Executors

    def _create_assigned_runner(self, op: AssignRunnerOp) -> AssignedRunner:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        assigned_runner = AssignedRunner(
            runner_id=op.runner_id,
            instance_id=op.instance_id,
            shard_metadata=op.shard_metadata,
            hosts=op.hosts,
            status=DownloadingRunnerStatus(
                download_progress=DownloadPending(node_id=self.node_id)
            ),
            runner=None,
        )
        self.assigned_runners[op.runner_id] = assigned_runner
        return assigned_runner

    async def _update_runner_status_to_completed_then_inactive(
        self, assigned_runner: AssignedRunner
    ) -> AsyncGenerator[Event, None]:
        """Updates runner status from downloading to completed, then to inactive."""
        assigned_runner.status = DownloadingRunnerStatus(
            download_progress=DownloadCompleted(node_id=self.node_id)
        )
        yield assigned_runner.status_update_event()

        assigned_runner.status = InactiveRunnerStatus()
        yield assigned_runner.status_update_event()

    async def _handle_already_downloaded_shard(
        self, assigned_runner: AssignedRunner
    ) -> AsyncGenerator[Event, None]:
        """Handles the case where the shard is already downloaded."""
        async for event in self._update_runner_status_to_completed_then_inactive(
            assigned_runner
        ):
            yield event

    async def _handle_shard_download_process(
        self,
        assigned_runner: AssignedRunner,
        op: AssignRunnerOp,
        initial_progress: RepoDownloadProgress,
    ) -> AsyncGenerator[Event, None]:
        """Manages the shard download process with progress tracking."""
        # Set initial ongoing status
        assigned_runner.status = DownloadingRunnerStatus(
            download_progress=DownloadOngoing(
                node_id=self.node_id,
                download_progress=DownloadProgressData(
                    total_bytes=initial_progress.total_bytes,
                    downloaded_bytes=initial_progress.downloaded_bytes,
                ),
            )
        )
        yield assigned_runner.status_update_event()

        # Set up download progress tracking
        download_progress_queue: asyncio.Queue[RepoDownloadProgress] = asyncio.Queue()

        def download_progress_callback(
            shard: ShardMetadata, progress: RepoDownloadProgress
        ) -> None:
            download_progress_queue.put_nowait(progress)

        self.shard_downloader.on_progress(download_progress_callback)
        download_task = asyncio.create_task(
            self.shard_downloader.ensure_shard(op.shard_metadata)
        )

        try:
            async for event in self._monitor_download_progress(
                assigned_runner, download_progress_queue
            ):
                yield event
        finally:
            if not download_task.done():
                download_task.cancel()

    async def _monitor_download_progress(
        self,
        assigned_runner: AssignedRunner,
        download_progress_queue: asyncio.Queue[RepoDownloadProgress],
    ) -> AsyncGenerator[Event, None]:
        """Monitors download progress and yields status updates."""
        last_progress_time = 0.0
        throttle_interval_secs = 1.0

        while True:
            progress: RepoDownloadProgress = await asyncio.wait_for(
                download_progress_queue.get(), timeout=15
            )

            if progress.status == "complete":
                async for (
                    event
                ) in self._update_runner_status_to_completed_then_inactive(
                    assigned_runner
                ):
                    yield event
                break
            elif progress.status == "in_progress":
                if time.monotonic() - last_progress_time > throttle_interval_secs:
                    assigned_runner.status = DownloadingRunnerStatus(
                        download_progress=DownloadOngoing(
                            node_id=self.node_id,
                            download_progress=DownloadProgressData(
                                total_bytes=progress.total_bytes,
                                downloaded_bytes=progress.downloaded_bytes,
                            ),
                        )
                    )
                    yield assigned_runner.status_update_event()
                    last_progress_time = time.monotonic()

    async def _execute_assign_op(
        self, op: AssignRunnerOp
    ) -> AsyncGenerator[Event, None]:
        """
        A runner has been assigned. We need to also ensure that it's downloaded.
        This op assigns the runner, and moves from Downloading -> Inactive (ready to spin) state.
        """
        assigned_runner = self._create_assigned_runner(op)
        initial_progress = (
            await self.shard_downloader.get_shard_download_status_for_shard(
                op.shard_metadata
            )
        )

        if initial_progress.status == "complete":
            async for event in self._handle_already_downloaded_shard(assigned_runner):
                yield event
        else:
            async for event in self._handle_shard_download_process(
                assigned_runner, op, initial_progress
            ):
                yield event

    async def _execute_unassign_op(
        self, op: UnassignRunnerOp
    ) -> AsyncGenerator[Event, None]:
        if op.runner_id not in self.assigned_runners:
            return

        # We can try to do a graceful shutdown of the runner.
        runner: RunnerSupervisor | None = self.assigned_runners[op.runner_id].runner
        if runner is not None:
            await runner.astop()

        # This is all we really need:
        del self.assigned_runners[op.runner_id]
        yield RunnerDeleted(runner_id=op.runner_id)

        return
        yield

    async def _execute_runner_up_op(
        self, op: RunnerUpOp, initialize_timeout: Optional[float] = None
    ) -> AsyncGenerator[Event, None]:
        assigned_runner = self.assigned_runners[op.runner_id]

        assigned_runner.runner = await RunnerSupervisor.create(
            model_shard_meta=assigned_runner.shard_metadata,
            hosts=assigned_runner.hosts,
            initialize_timeout=initialize_timeout,
        )

        if assigned_runner.runner.healthy:
            assigned_runner.status = LoadedRunnerStatus()
        else:
            # Log detailed reasons why the runner is not healthy
            runner = assigned_runner.runner
            health_issues: list[str] = []

            if runner.runner_process.returncode is not None:
                health_issues.append(
                    f"runner_process.returncode is {runner.runner_process.returncode}"
                )
            if runner.runner_process.stdin is None:
                health_issues.append("runner_process.stdin is None")
            elif runner.runner_process.stdin.is_closing():
                health_issues.append("runner_process.stdin is closing")
            if runner.runner_process.stdout is None:
                health_issues.append("runner_process.stdout is None")

            logger.warning(f"Runner status is not healthy: {', '.join(health_issues)}")
            assigned_runner.status = FailedRunnerStatus()
        yield self.assigned_runners[op.runner_id].status_update_event()

    async def _execute_runner_down_op(
        self, op: RunnerDownOp
    ) -> AsyncGenerator[Event, None]:
        assigned_runner = self.assigned_runners[op.runner_id]

        if isinstance(assigned_runner.runner, RunnerSupervisor):
            await assigned_runner.runner.astop()

        assigned_runner.runner = None

        assigned_runner.status = InactiveRunnerStatus()
        yield assigned_runner.status_update_event()
        return

    async def _execute_runner_failed_op(
        self, op: RunnerFailedOp
    ) -> AsyncGenerator[Event, None]:
        """
        We detected that this runner has failed. So we'll put it into 'failed' state now, triggering the rest of the instance to spin down.
        """
        assigned_runner = self.assigned_runners[op.runner_id]

        if isinstance(assigned_runner.runner, RunnerSupervisor):
            await (
                assigned_runner.runner.astop()
            )  # astop the runner to ensure it clears out of memory.

        assigned_runner.status = FailedRunnerStatus()
        yield self.assigned_runners[op.runner_id].status_update_event()

    async def _execute_task_op(self, op: ExecuteTaskOp) -> AsyncGenerator[Event, None]:
        """
        This is the entry point for a chat completion starting.
        While there is only one execute function, it will get called in different ways for runner 0 and runner [1, 2, 3, ...].
        Runners [1, 2, 3, ...] will run this method when a task is in 'pending' state.
        Runner 0 will run this method when a task is in 'running' state.
        TODO: How do we handle the logic of ensuring that n-1 nodes have started their execution before allowing the 0'th runner to start?
        This is still a little unclear to me.
        """
        assigned_runner = self.assigned_runners[op.runner_id]

        async def inner_execute(queue: asyncio.Queue[Event]) -> None:
            async def running_callback(queue: asyncio.Queue[Event]) -> None:
                # Called when the MLX process has been kicked off
                assigned_runner.status = RunningRunnerStatus()
                await queue.put(assigned_runner.status_update_event())

                if assigned_runner.shard_metadata.device_rank == 0:
                    await queue.put(
                        TaskStateUpdated(
                            task_id=op.task.task_id,
                            task_status=TaskStatus.RUNNING,
                        )
                    )

            assert assigned_runner.runner is not None
            assert assigned_runner.runner.healthy

            async for chunk in assigned_runner.runner.stream_response(
                task=op.task, request_started_callback=partial(running_callback, queue)
            ):
                if assigned_runner.shard_metadata.device_rank == 0:
                    await queue.put(
                        ChunkGenerated(
                            # todo: at some point we will no longer have a bijection between task_id and row_id.
                            # So we probably want to store a mapping between these two in our Worker object.
                            command_id=chunk.command_id,
                            chunk=chunk,
                        )
                    )

            if assigned_runner.shard_metadata.device_rank == 0:
                await queue.put(
                    TaskStateUpdated(
                        task_id=op.task.task_id,
                        task_status=TaskStatus.COMPLETE,
                    )
                )

            # After a successful inference:
            assigned_runner.status = LoadedRunnerStatus()
            await queue.put(assigned_runner.status_update_event())

        queue: Queue[Event] = asyncio.Queue()
        task = asyncio.create_task(inner_execute(queue))

        # TODO: Initial (prefil) timeout can be dynamic
        # model_kb = assigned_runner.shard_metadata.model_meta.storage_size_kilobytes

        try:
            # Yield items from the queue
            while True:
                if task.done() and (exception := task.exception()):
                    raise exception

                try:
                    # Use a timeout to periodically check task status
                    item: Event = await asyncio.wait_for(queue.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    continue

                yield item
                if isinstance(item, RunnerStatusUpdated) and isinstance(
                    item.runner_status, (LoadedRunnerStatus, FailedRunnerStatus)
                ):
                    if isinstance(item.runner_status, LoadedRunnerStatus):
                        assigned_runner.failures = []

                    break
        finally:
            # Ensure the task is cleaned up
            try:
                await asyncio.wait_for(task, timeout=5)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out waiting for task cleanup after inference execution."
                )

    ## Operation Planner

    async def execute_op(self, op: RunnerOp) -> AsyncGenerator[Event, None]:
        ## It would be great if we can get rid of this async for ... yield pattern.
        match op.op_type:
            case RunnerOpType.ASSIGN_RUNNER:
                event_generator = self._execute_assign_op(op)
            case RunnerOpType.UNASSIGN_RUNNER:
                event_generator = self._execute_unassign_op(op)
            case RunnerOpType.RUNNER_UP:
                event_generator = self._execute_runner_up_op(op)
            case RunnerOpType.RUNNER_DOWN:
                event_generator = self._execute_runner_down_op(op)
            case RunnerOpType.RUNNER_FAILED:
                event_generator = self._execute_runner_failed_op(op)
            case RunnerOpType.CHAT_COMPLETION:
                event_generator = self._execute_task_op(op)

        async for event in event_generator:
            yield event

    async def fail_runner(
        self, e: Exception, runner_id: RunnerId
    ) -> AsyncGenerator[Event]:
        if runner_id in self.assigned_runners:
            assigned_runner = self.assigned_runners[runner_id]

            assigned_runner.runner = None
            assigned_runner.status = FailedRunnerStatus(error_message=str(e))
            assigned_runner.failures.append((time.time(), e))

            # Reset failure count back to 0 when succesful
            if len(assigned_runner.failures) >= 3:
                # Too many retries. We will emit a DeleteInstance
                yield InstanceDeleted(instance_id=assigned_runner.instance_id)

            yield assigned_runner.status_update_event()

    async def fail_task(
        self, e: Exception, runner_id: RunnerId, task_id: TaskId
    ) -> AsyncGenerator[Event]:
        if runner_id in self.assigned_runners:
            yield TaskStateUpdated(
                task_id=task_id,
                task_status=TaskStatus.FAILED,
            )

            yield TaskFailed(
                task_id=task_id, error_type=str(type(e)), error_message=str(e)
            )

            async for event in self.fail_runner(e, runner_id):
                yield event

    async def event_publisher(self, event: Event) -> None:
        assert self.worker_events is not None
        await self.worker_events.append_events([event], self.node_id)
        logger.info(f"published event: {event}")
