import asyncio
import time
from asyncio import Queue
from functools import partial
from random import random
from typing import AsyncGenerator, Optional

import anyio
from anyio import CancelScope, create_task_group
from anyio.abc import TaskGroup
from loguru import logger

from exo.routing.connection_message import ConnectionMessage, ConnectionMessageType
from exo.shared.apply import apply
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    EventId,
    ForwarderEvent,
    IndexedEvent,
    InstanceDeleted,
    NodeMemoryMeasured,
    NodePerformanceMeasured,
    RunnerDeleted,
    RunnerStatusUpdated,
    TaskFailed,
    TaskStateUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import MemoryPerformanceProfile, NodePerformanceProfile
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus
from exo.shared.types.topology import Connection
from exo.shared.types.worker.common import RunnerId
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadPending,
)
from exo.shared.types.worker.ops import (
    AssignRunnerOp,
    ExecuteTaskOp,
    RunnerDownOp,
    RunnerFailedOp,
    RunnerOp,
    RunnerUpOp,
    UnassignRunnerOp,
)
from exo.shared.types.worker.runners import (
    DownloadingRunnerStatus,
    FailedRunnerStatus,
    InactiveRunnerStatus,
    LoadedRunnerStatus,
    RunningRunnerStatus,
    StartingRunnerStatus,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import Receiver, Sender
from exo.utils.event_buffer import OrderedBuffer
from exo.worker.common import AssignedRunner
from exo.worker.download.download_utils import (
    map_repo_download_progress_to_download_progress_data,
)
from exo.worker.download.shard_downloader import RepoDownloadProgress, ShardDownloader
from exo.worker.plan import plan
from exo.worker.runner.runner_supervisor import RunnerSupervisor
from exo.worker.utils import start_polling_memory_metrics, start_polling_node_metrics


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        shard_downloader: ShardDownloader,
        *,
        initial_connection_messages: list[ConnectionMessage],
        connection_message_receiver: Receiver[ConnectionMessage],
        # Having written this pattern 3 times in the codebase:
        # Should this be inherited??? Is this a real inheritance
        # W????
        # Limitation: This SHOULD be a MasterForwarderEvent, but inheritance says no :|
        global_event_receiver: Receiver[ForwarderEvent],
        # Limitation: This SHOULD be a WorkerForwarderEvent, but inheritance says no :|
        local_event_sender: Sender[ForwarderEvent],
        # This is for requesting updates. It doesn't need to be a general command sender right now,
        # but I think it's the correct way to be thinking about commands
        command_sender: Sender[ForwarderCommand],
    ):
        self.node_id: NodeId = node_id
        self.shard_downloader: ShardDownloader = shard_downloader
        self.global_event_receiver = global_event_receiver
        self.local_event_sender = local_event_sender
        self.local_event_index = 0
        self.command_sender = command_sender
        self.connection_message_receiver = connection_message_receiver
        self.event_buffer = OrderedBuffer[Event]()
        self._initial_connection_messages = initial_connection_messages
        self.out_for_delivery: dict[EventId, ForwarderEvent] = {}

        self.state: State = State()
        self.assigned_runners: dict[RunnerId, AssignedRunner] = {}
        self._tg: TaskGroup | None = None
        self._nack_cancel_scope: CancelScope | None = None

    async def run(self):
        logger.info("Starting Worker")

        # TODO: CLEANUP HEADER
        async def resource_monitor_callback(
            node_performance_profile: NodePerformanceProfile,
        ) -> None:
            await self.event_publisher(
                NodePerformanceMeasured(
                    node_id=self.node_id, node_profile=node_performance_profile
                ),
            )

        async def memory_monitor_callback(
            memory_profile: MemoryPerformanceProfile,
        ) -> None:
            await self.event_publisher(
                NodeMemoryMeasured(node_id=self.node_id, memory=memory_profile)
            )

        # END CLEANUP

        async with create_task_group() as tg:
            self._tg = tg
            tg.start_soon(start_polling_node_metrics, resource_monitor_callback)

            tg.start_soon(start_polling_memory_metrics, memory_monitor_callback)
            tg.start_soon(self._connection_message_event_writer)
            tg.start_soon(self._resend_out_for_delivery)
            tg.start_soon(self._event_applier)
            # TODO: This is a little gross, but not too bad
            for msg in self._initial_connection_messages:
                await self.event_publisher(
                    self._convert_connection_message_to_event(msg)
                )
            self._initial_connection_messages = []

        # Actual shutdown code - waits for all tasks to complete before executing.
        self.local_event_sender.close()
        self.command_sender.close()
        for runner in self.assigned_runners.values():
            if runner.runner:
                await runner.runner.astop()

    async def _event_applier(self):
        with self.global_event_receiver as events:
            async for event in events:
                self.event_buffer.ingest(event.origin_idx, event.event)
                event_id = event.event.event_id
                if event_id in self.out_for_delivery:
                    del self.out_for_delivery[event_id]

                # 2. for each event, apply it to the state
                indexed_events = self.event_buffer.drain_indexed()
                if not indexed_events:
                    if (
                        self._nack_cancel_scope is None
                        or self._nack_cancel_scope.cancel_called
                    ):
                        assert self._tg
                        self._tg.start_soon(self._nack_request)
                elif self._nack_cancel_scope:
                    self._nack_cancel_scope.cancel()

                flag = False
                for idx, event in indexed_events:
                    self.state = apply(self.state, IndexedEvent(idx=idx, event=event))
                    if event_relevant_to_worker(event, self):
                        flag = True

                # 3. If we've found a "relevant" event, run a plan -> op -> execute cycle.
                if flag:
                    await self.plan_step()

    async def plan_step(self):
        # 3. based on the updated state, we plan & execute an operation.
        op: RunnerOp | None = plan(
            self.assigned_runners,
            self.node_id,
            self.state.instances,
            self.state.runners,
            self.state.tasks,
        )

        # run the op, synchronously blocking for now
        if op is not None:
            logger.info(f"Executing op {str(op)[:100]}")
            logger.debug(f"Worker executing op: {str(op)[:100]}")
            try:
                async for event in self.execute_op(op):
                    await self.event_publisher(event)
            except Exception as e:
                if isinstance(op, ExecuteTaskOp):
                    generator = self.fail_task(
                        e, runner_id=op.runner_id, task_id=op.task.task_id
                    )
                else:
                    generator = self.fail_runner(e, runner_id=op.runner_id)

                async for event in generator:
                    await self.event_publisher(event)

    def shutdown(self):
        if self._tg:
            self._tg.cancel_scope.cancel()

    async def _connection_message_event_writer(self):
        with self.connection_message_receiver as connection_messages:
            async for msg in connection_messages:
                await self.event_publisher(
                    self._convert_connection_message_to_event(msg)
                )

    def _convert_connection_message_to_event(self, msg: ConnectionMessage):
        match msg.connection_type:
            case ConnectionMessageType.Connected:
                return TopologyEdgeCreated(
                    edge=Connection(
                        local_node_id=self.node_id,
                        send_back_node_id=msg.node_id,
                        send_back_multiaddr=Multiaddr(
                            address=f"/ip4/{msg.remote_ipv4}/tcp/{msg.remote_tcp_port}"
                        ),
                    )
                )

            case ConnectionMessageType.Disconnected:
                return TopologyEdgeDeleted(
                    edge=Connection(
                        local_node_id=self.node_id,
                        send_back_node_id=msg.node_id,
                        send_back_multiaddr=Multiaddr(
                            address=f"/ip4/{msg.remote_ipv4}/tcp/{msg.remote_tcp_port}"
                        ),
                    )
                )

    async def _nack_request(self) -> None:
        # This function is started whenever we receive an event that is out of sequence.
        # It is cancelled as soon as we receiver an event that is in sequence.
        # Thus, if we don't make any progress within 1 + random() seconds, we request a copy of the event log
        # This can be MASSIVELY tightened - just requesting a single event should be sufficient.
        with CancelScope() as scope:
            self._nack_cancel_scope = scope
            try:
                await anyio.sleep(1 + random())
                await self.command_sender.send(
                    ForwarderCommand(
                        origin=self.node_id,
                        command=RequestEventLog(since_idx=0),
                    )
                )
            finally:
                if self._nack_cancel_scope is scope:
                    self._nack_cancel_scope = None

    async def _resend_out_for_delivery(self) -> None:
        # This can also be massively tightened, we should check events are at least a certain age before resending.
        # Exponential backoff would also certainly help here.
        while True:
            await anyio.sleep(1 + random())
            for event in self.out_for_delivery.copy().values():
                await self.local_event_sender.send(event)

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
                download_progress=map_repo_download_progress_to_download_progress_data(initial_progress),
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
                            download_progress=map_repo_download_progress_to_download_progress_data(progress),
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

    async def _execute_runner_up_op(
        self, op: RunnerUpOp, initialize_timeout: Optional[float] = None
    ) -> AsyncGenerator[Event, None]:
        assigned_runner = self.assigned_runners[op.runner_id]

        # Emit "Starting" status right away so UI can show loading state
        assigned_runner.status = StartingRunnerStatus()
        yield assigned_runner.status_update_event()

        assigned_runner.runner = await RunnerSupervisor.create(
            model_shard_meta=assigned_runner.shard_metadata,
            hosts=assigned_runner.hosts,
            initialize_timeout=initialize_timeout,
        )

        if assigned_runner.runner.runner_process.is_alive():
            assigned_runner.status = LoadedRunnerStatus()
        else:
            runner = assigned_runner.runner
            logger.warning(
                f"Runner status is not runner_process.is_alive(): exit code {runner.runner_process.exitcode}"
            )

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
                            task_status=TaskStatus.Running,
                        )
                    )

            assert assigned_runner.runner is not None
            assert assigned_runner.runner.runner_process.is_alive()

            async for chunk in assigned_runner.runner.stream_response(
                task=op.task, request_started_callback=partial(running_callback, queue)
            ):
                if assigned_runner.shard_metadata.device_rank == 0:
                    await queue.put(
                        ChunkGenerated(
                            # TODO: at some point we will no longer have a bijection between task_id and row_id.
                            # So we probably want to store a mapping between these two in our Worker object.
                            command_id=chunk.command_id,
                            chunk=chunk,
                        )
                    )

            if op.task.task_id in self.state.tasks:
                self.state.tasks[op.task.task_id].task_status = TaskStatus.Complete

            if assigned_runner.shard_metadata.device_rank == 0:
                # kind of hack - we don't want to wait for the round trip for this to complete
                await queue.put(
                    TaskStateUpdated(
                        task_id=op.task.task_id,
                        task_status=TaskStatus.Complete,
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
        match op:
            case AssignRunnerOp():
                event_generator = self._execute_assign_op(op)
            case UnassignRunnerOp():
                event_generator = self._execute_unassign_op(op)
            case RunnerUpOp():
                event_generator = self._execute_runner_up_op(op)
            case RunnerDownOp():
                event_generator = self._execute_runner_down_op(op)
            case RunnerFailedOp():
                event_generator = self._execute_runner_failed_op(op)
            case ExecuteTaskOp():
                event_generator = self._execute_task_op(op)

        async for event in event_generator:
            yield event

    async def fail_runner(
        self, e: Exception, runner_id: RunnerId
    ) -> AsyncGenerator[Event]:
        if runner_id in self.assigned_runners:
            assigned_runner = self.assigned_runners[runner_id]

            if assigned_runner.runner is not None:
                await assigned_runner.runner.astop()
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
                task_status=TaskStatus.Failed,
            )

            yield TaskFailed(
                task_id=task_id, error_type=str(type(e)), error_message=str(e)
            )

            async for event in self.fail_runner(e, runner_id):
                yield event



    # This function is re-entrant, take care!
    async def event_publisher(self, event: Event) -> None:
        fe = ForwarderEvent(
            origin_idx=self.local_event_index,
            origin=self.node_id,
            event=event,
        )
        logger.debug(
            f"Worker published event {self.local_event_index}: {str(event)[:100]}"
        )
        self.local_event_index += 1
        await self.local_event_sender.send(fe)
        self.out_for_delivery[event.event_id] = fe


def event_relevant_to_worker(event: Event, worker: Worker):
    # TODO
    return True
