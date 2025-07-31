import asyncio
import logging
import time
from asyncio import Queue
from functools import partial
from time import process_time
from typing import AsyncGenerator, Optional

from shared.db.sqlite import AsyncSQLiteEventStorage
from shared.types.common import NodeId
from shared.types.events import (
    ChunkGenerated,
    Event,
    InstanceDeleted,
    RunnerDeleted,
    RunnerStatusUpdated,
    TaskFailed,
    TaskStateUpdated,
)
from shared.types.state import State
from shared.types.tasks import TaskId, TaskStatus
from shared.types.worker.common import RunnerId
from shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgressData,
)
from shared.types.worker.ops import (
    AssignRunnerOp,
    ExecuteTaskOp,
    RunnerDownOp,
    RunnerFailedOp,
    RunnerOp,
    RunnerOpType,
    RunnerUpOp,
    UnassignRunnerOp,
)
from shared.types.worker.runners import (
    DownloadingRunnerStatus,
    FailedRunnerStatus,
    InactiveRunnerStatus,
    LoadedRunnerStatus,
    RunningRunnerStatus,
)
from shared.types.worker.shards import ShardMetadata
from worker.common import AssignedRunner
from worker.download.shard_downloader import RepoDownloadProgress, ShardDownloader
from worker.runner.runner_supervisor import RunnerSupervisor


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        logger: logging.Logger,
        shard_downloader: ShardDownloader,
        worker_events: AsyncSQLiteEventStorage | None,
        global_events: AsyncSQLiteEventStorage | None,
    ):
        self.node_id: NodeId = node_id
        self.state: State = State()
        self.shard_downloader: ShardDownloader = shard_downloader
        self.worker_events: AsyncSQLiteEventStorage | None = worker_events # worker_events is None in some tests.
        self.global_events: AsyncSQLiteEventStorage | None = global_events
        self.logger: logging.Logger = logger

        self.assigned_runners: dict[RunnerId, AssignedRunner] = {}
        self._task: asyncio.Task[None] | None = None

    ## Op Executors

    async def _execute_assign_op(
        self, op: AssignRunnerOp
    ) -> AsyncGenerator[Event, None]:
        '''
        A runner has been assigned. We need to also ensure that it's downloaded.
        This op assigns the runner, and moves from Downloading -> Inactive (ready to spin) state.
        '''
        self.assigned_runners[op.runner_id] = AssignedRunner(
            runner_id=op.runner_id,
            instance_id=op.instance_id,
            shard_metadata=op.shard_metadata,
            hosts=op.hosts,
            status=DownloadingRunnerStatus(
                download_progress=DownloadPending(
                    node_id=self.node_id
                )
            ),
            runner=None,
        )

        assigned_runner = self.assigned_runners[op.runner_id]
        initial_progress = await self.shard_downloader.get_shard_download_status_for_shard(op.shard_metadata)

        if initial_progress.status == "complete":
            assigned_runner.status = DownloadingRunnerStatus(
                download_progress=DownloadCompleted(
                    node_id=self.node_id
                )
            )
            yield assigned_runner.status_update_event()

            assigned_runner.status = InactiveRunnerStatus()
            yield assigned_runner.status_update_event()

            return
        else:
            assigned_runner.status = DownloadingRunnerStatus(
                download_progress=DownloadOngoing(
                    node_id=self.node_id,
                    download_progress=DownloadProgressData(
                        total_bytes=initial_progress.total_bytes,
                        downloaded_bytes=initial_progress.downloaded_bytes
                    )
                )
            )
            yield assigned_runner.status_update_event()

            # Download it!
            # TODO: we probably want download progress as part of a callback that gets passed to the downloader.
            download_progress_queue: asyncio.Queue[RepoDownloadProgress] = asyncio.Queue()
            def download_progress_callback(shard: ShardMetadata, progress: RepoDownloadProgress) -> None:
                download_progress_queue.put_nowait(progress)


            self.shard_downloader.on_progress(download_progress_callback)

            asyncio.create_task(self.shard_downloader.ensure_shard(op.shard_metadata))

            # TODO: Dynamic timeout, timeout on no packet update received.
            timeout_secs = 10 * 60
            start_time = process_time()
            last_yield_progress = start_time
            while process_time() - start_time < timeout_secs:
                progress: RepoDownloadProgress = await download_progress_queue.get()
                if progress.status == "complete":
                    assigned_runner.status = DownloadingRunnerStatus(
                        download_progress=DownloadCompleted(
                            node_id=self.node_id,
                        )
                    )
                    yield assigned_runner.status_update_event()

                    assigned_runner.status = InactiveRunnerStatus()
                    yield assigned_runner.status_update_event()

                    break
                elif progress.status == "in_progress":
                    if process_time() - last_yield_progress > 1:
                        assigned_runner.status = DownloadingRunnerStatus(
                            download_progress=DownloadOngoing(
                                node_id=self.node_id,
                                download_progress=DownloadProgressData(
                                    total_bytes=progress.total_bytes,
                                    downloaded_bytes=progress.downloaded_bytes,
                                )
                            )
                        )
                        yield assigned_runner.status_update_event()

                        last_yield_progress = process_time()
            else:
                assigned_runner.status = DownloadingRunnerStatus(
                    download_progress=DownloadFailed(
                        node_id=self.node_id,
                        error_message=f"Timeout downloading model: {op.shard_metadata.model_meta.model_id}"
                    )
                )
                yield assigned_runner.status_update_event()

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

        # TODO: This should be dynamic, based on the size of the model.
        if not initialize_timeout:
            gigabytes_per_second = 10
            kilobytes_per_second = gigabytes_per_second * 1024 * 1024
            
            shard = assigned_runner.shard_metadata
            weights_size_kb = (shard.end_layer - shard.start_layer) / shard.n_layers * shard.model_meta.storage_size_kilobytes
            
            initialize_timeout = weights_size_kb / kilobytes_per_second + 120.0 # Add a constant 120.0 to ensure connection can be made as well

            self.logger.info(f"initialize_timeout: {initialize_timeout}")

        try:
            assigned_runner.runner = await asyncio.wait_for(
                RunnerSupervisor.create(
                    model_shard_meta=assigned_runner.shard_metadata,
                    hosts=assigned_runner.hosts,
                    logger=self.logger,
                ),
                timeout=initialize_timeout,
            )
        except TimeoutError as e:
            import traceback

            tb = traceback.format_exc()
            e = Exception(f"{type(e).__name__}: {str(e)}. Traceback: {tb}")
            async for event in self._fail_runner(e=e, runner_id=op.runner_id):
                yield event
            return

        if assigned_runner.runner.healthy:
            assigned_runner.status = LoadedRunnerStatus()
        else:
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
        '''
        We detected that this runner has failed. So we'll put it into 'failed' state now, triggering the rest of the instance to spin down.
        '''
        assigned_runner = self.assigned_runners[op.runner_id]

        assigned_runner.status = FailedRunnerStatus()
        yield self.assigned_runners[op.runner_id].status_update_event()


    async def _execute_task_op(
        self, op: ExecuteTaskOp
    ) -> AsyncGenerator[Event, None]:
        '''
        This is the entry point for a chat completion starting.
        While there is only one execute function, it will get called in different ways for runner 0 and runner [1, 2, 3, ...].
        Runners [1, 2, 3, ...] will run this method when a task is in 'pending' state.
        Runner 0 will run this method when a task is in 'running' state.
        TODO: How do we handle the logic of ensuring that n-1 nodes have started their execution before allowing the 0'th runner to start?
        This is still a little unclear to me.
        '''
        assigned_runner = self.assigned_runners[op.runner_id]

        async def inner_execute(queue: asyncio.Queue[Event]) -> None:
            async def running_callback(queue: asyncio.Queue[Event]) -> None:
                # Called when the MLX process has been kicked off
                assigned_runner.status = RunningRunnerStatus()
                await queue.put(assigned_runner.status_update_event())

                if assigned_runner.shard_metadata.device_rank == 0:
                    await queue.put(TaskStateUpdated(
                        task_id=op.task.task_id,
                        task_status=TaskStatus.RUNNING,
                    ))

            try:
                assert assigned_runner.runner is not None
                assert assigned_runner.runner.healthy

                async for chunk in assigned_runner.runner.stream_response(
                        task=op.task,
                        request_started_callback=partial(running_callback, queue)):
                    if assigned_runner.shard_metadata.device_rank == 0:
                        await queue.put(ChunkGenerated(
                            # todo: at some point we will no longer have a bijection between task_id and row_id. 
                            # So we probably want to store a mapping between these two in our Worker object.
                            command_id=chunk.command_id, 
                            chunk=chunk
                        ))

                if assigned_runner.shard_metadata.device_rank == 0:
                    await queue.put(TaskStateUpdated(
                        task_id=op.task.task_id,
                        task_status=TaskStatus.COMPLETE,
                    ))

                # After a successful inference:
                assigned_runner.status = LoadedRunnerStatus()
                await queue.put(assigned_runner.status_update_event())


            except Exception as e:
                # An exception occurs in the runner supervisor
                self.logger.warning(f'Runner failed whilst running inference task. Task: {op.task}. Error: {e}')
                async for event in self._fail_task(e, op.runner_id, op.task.task_id):
                    await queue.put(event)

        queue: Queue[Event] = asyncio.Queue()
        task = asyncio.create_task(inner_execute(queue))

        # TODO: Initial (prefil) timeout can be dynamic
        # model_kb = assigned_runner.shard_metadata.model_meta.storage_size_kilobytes

        try:
            # Yield items from the queue
            # timeout = 30.
            timeout = 3.
            while True:
                item: Event = await asyncio.wait_for(queue.get(), timeout=timeout)
                yield item
                timeout = 2.
                if isinstance(item, RunnerStatusUpdated) and isinstance(
                    item.runner_status, (LoadedRunnerStatus, FailedRunnerStatus)
                ):
                    if isinstance(item.runner_status, LoadedRunnerStatus):
                        assigned_runner.failures = []
                        
                    break
        except TimeoutError as e:
            # Runner supervisor doesn't respond in time; so we put the runner & task into a failed state
            self.logger.warning(f'Timed out waiting for runner response to inference task. Task: {op.task}.')
            async for event in self._fail_task(e, op.runner_id, op.task.task_id):
                yield event
        finally:
            # Ensure the task is cleaned up
            try:
                await asyncio.wait_for(task, timeout=5)
            except asyncio.TimeoutError:
                self.logger.warning("Timed out waiting for task cleanup after inference execution.")
        

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


    async def _fail_runner(self, e: Exception, runner_id: RunnerId) -> AsyncGenerator[Event]:
        if runner_id in self.assigned_runners:
            assigned_runner = self.assigned_runners[runner_id]

            assigned_runner.runner = None
            assigned_runner.status = FailedRunnerStatus(error_message=str(e))
            assigned_runner.failures.append(
                (
                    time.time(),
                    e
                )
            )

            # Reset failure count back to 0 when succesful
            if len(assigned_runner.failures) >= 3:
                # Too many retries. We will emit a DeleteInstance 
                yield InstanceDeleted(
                    instance_id=assigned_runner.instance_id
                )

            yield assigned_runner.status_update_event()

    
    async def _fail_task(self, e: Exception, runner_id: RunnerId, task_id: TaskId) -> AsyncGenerator[Event]:
        if runner_id in self.assigned_runners:
            yield TaskStateUpdated(
                task_id=task_id,
                task_status=TaskStatus.FAILED,
            )

            yield TaskFailed(
                task_id=task_id,
                error_type=str(type(e)),
                error_message=str(e)
            )

            async for event in self._fail_runner(e, runner_id):
                yield event


    async def event_publisher(self, event: Event) -> None:
        assert self.worker_events is not None
        await self.worker_events.append_events([event], self.node_id)
        self.logger.info(f"published event: {event}")

