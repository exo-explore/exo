from random import random

import anyio
from anyio import CancelScope, create_task_group, current_time, fail_after
from anyio.abc import TaskGroup
from loguru import logger

from exo.routing.connection_message import ConnectionMessage, ConnectionMessageType
from exo.shared.apply import apply
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    Event,
    EventId,
    ForwarderEvent,
    IndexedEvent,
    NodeDownloadProgress,
    NodeMemoryMeasured,
    NodePerformanceMeasured,
    TaskCreated,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import MemoryPerformanceProfile, NodePerformanceProfile
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CreateRunner,
    DownloadModel,
    Shutdown,
    Task,
    TaskStatus,
)
from exo.shared.types.topology import Connection
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.event_buffer import OrderedBuffer
from exo.worker.download.download_utils import (
    map_repo_download_progress_to_download_progress_data,
)
from exo.worker.download.shard_downloader import RepoDownloadProgress, ShardDownloader
from exo.worker.plan import plan
from exo.worker.runner.runner_supervisor import RunnerSupervisor
from exo.worker.utils import start_polling_memory_metrics, start_polling_node_metrics
from exo.worker.utils.net_profile import connect_all


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        shard_downloader: ShardDownloader,
        *,
        initial_connection_messages: list[ConnectionMessage],
        connection_message_receiver: Receiver[ConnectionMessage],
        global_event_receiver: Receiver[ForwarderEvent],
        local_event_sender: Sender[ForwarderEvent],
        # This is for requesting updates. It doesn't need to be a general command sender right now,
        # but I think it's the correct way to be thinking about commands
        command_sender: Sender[ForwarderCommand],
    ):
        self.node_id: NodeId = node_id
        self.session_id: SessionId = session_id

        self.shard_downloader: ShardDownloader = shard_downloader
        self._pending_downloads: dict[RunnerId, ShardMetadata] = {}

        self.global_event_receiver = global_event_receiver
        self.local_event_sender = local_event_sender
        self.local_event_index = 0
        self.command_sender = command_sender
        self.connection_message_receiver = connection_message_receiver
        self.event_buffer = OrderedBuffer[Event]()
        self._initial_connection_messages = initial_connection_messages
        self.out_for_delivery: dict[EventId, ForwarderEvent] = {}

        self.state: State = State()
        self.download_status: dict[ShardMetadata, DownloadProgress] = {}
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup | None = None
        self._nack_cancel_scope: CancelScope | None = None

        self.event_sender, self.event_receiver = channel[Event]()

    async def run(self):
        logger.info("Starting Worker")

        # TODO: CLEANUP HEADER
        async def resource_monitor_callback(
            node_performance_profile: NodePerformanceProfile,
        ) -> None:
            await self.event_sender.send(
                NodePerformanceMeasured(
                    node_id=self.node_id, node_profile=node_performance_profile
                ),
            )

        async def memory_monitor_callback(
            memory_profile: MemoryPerformanceProfile,
        ) -> None:
            await self.event_sender.send(
                NodeMemoryMeasured(node_id=self.node_id, memory=memory_profile)
            )

        # END CLEANUP

        async with create_task_group() as tg:
            self._tg = tg
            tg.start_soon(self.plan_step)
            tg.start_soon(start_polling_node_metrics, resource_monitor_callback)

            tg.start_soon(start_polling_memory_metrics, memory_monitor_callback)
            tg.start_soon(self._connection_message_event_writer)
            tg.start_soon(self._resend_out_for_delivery)
            tg.start_soon(self._event_applier)
            tg.start_soon(self._forward_events)
            tg.start_soon(self._poll_connection_updates)
            # TODO: This is a little gross, but not too bad
            for msg in self._initial_connection_messages:
                await self.event_sender.send(
                    self._convert_connection_message_to_event(msg)
                )
            self._initial_connection_messages = []

        # Actual shutdown code - waits for all tasks to complete before executing.
        self.local_event_sender.close()
        self.command_sender.close()
        for runner in self.runners.values():
            runner.shutdown()

    async def _event_applier(self):
        with self.global_event_receiver as events:
            async for event in events:
                self.event_buffer.ingest(event.origin_idx, event.event)
                event_id = event.event.event_id
                if event_id in self.out_for_delivery:
                    del self.out_for_delivery[event_id]

                # 2. for each event, apply it to the state
                indexed_events = self.event_buffer.drain_indexed()
                if not indexed_events and (
                    self._nack_cancel_scope is None
                    or self._nack_cancel_scope.cancel_called
                ):
                    assert self._tg
                    self._tg.start_soon(self._nack_request)
                elif indexed_events and self._nack_cancel_scope:
                    self._nack_cancel_scope.cancel()

                flag = False
                for idx, event in indexed_events:
                    self.state = apply(self.state, IndexedEvent(idx=idx, event=event))
                    if event_relevant_to_worker(event, self):
                        flag = True

                # 3. If we've found a "relevant" event, run a plan -> op -> execute cycle.
                if flag:
                    # await self.plan_step()
                    pass

    async def plan_step(self):
        while True:
            await anyio.sleep(0.1)
            # 3. based on the updated state, we plan & execute an operation.
            task: Task | None = plan(
                self.node_id,
                self.runners,
                self.download_status,
                self.state.downloads,
                self.state.instances,
                self.state.runners,
                self.state.tasks,
            )
            if task is None:
                continue
            logger.info(f"Worker plan: {task.__class__.__name__}")
            assert task.task_status
            await self.event_sender.send(TaskCreated(task_id=task.task_id, task=task))

            # lets not kill the worker if a runner is unresponsive
            match task:
                case CreateRunner():
                    self._create_supervisor(task)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case DownloadModel(shard_metadata=shard):
                    if shard not in self.download_status:
                        progress = DownloadPending(
                            shard_metadata=shard, node_id=self.node_id
                        )
                        self.download_status[shard] = progress
                        await self.event_sender.send(
                            NodeDownloadProgress(download_progress=progress)
                        )
                    initial_progress = (
                        await self.shard_downloader.get_shard_download_status_for_shard(
                            shard
                        )
                    )
                    if initial_progress.status == "complete":
                        progress = DownloadCompleted(
                            shard_metadata=shard, node_id=self.node_id
                        )
                        self.download_status[shard] = progress
                        await self.event_sender.send(
                            NodeDownloadProgress(download_progress=progress)
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Complete,
                            )
                        )
                    else:
                        self.event_sender.send_nowait(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.Running
                            )
                        )
                        self._handle_shard_download_process(task, initial_progress)
                case Shutdown(runner_id=runner_id):
                    try:
                        with fail_after(3):
                            await self.runners.pop(runner_id).start_task(task)
                    except TimeoutError:
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.TimedOut
                            )
                        )
                case task:
                    await self.runners[self._task_to_runner_id(task)].start_task(task)

    def shutdown(self):
        if self._tg:
            self._tg.cancel_scope.cancel()

    def _task_to_runner_id(self, task: Task):
        instance = self.state.instances[task.instance_id]
        return instance.shard_assignments.node_to_runner[self.node_id]

    async def _connection_message_event_writer(self):
        with self.connection_message_receiver as connection_messages:
            async for msg in connection_messages:
                await self.event_sender.send(
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

    def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        runner = RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
        assert self._tg
        self._tg.start_soon(runner.run)
        return runner

    def _handle_shard_download_process(
        self,
        task: DownloadModel,
        initial_progress: RepoDownloadProgress,
    ):
        """Manages the shard download process with progress tracking."""
        status = DownloadOngoing(
            node_id=self.node_id,
            shard_metadata=task.shard_metadata,
            download_progress=map_repo_download_progress_to_download_progress_data(
                initial_progress
            ),
        )
        self.download_status[task.shard_metadata] = status
        self.event_sender.send_nowait(NodeDownloadProgress(download_progress=status))

        last_progress_time = 0.0
        throttle_interval_secs = 1.0

        # TODO: i hate callbacks
        def download_progress_callback(
            shard: ShardMetadata, progress: RepoDownloadProgress
        ) -> None:
            nonlocal self
            nonlocal last_progress_time
            if progress.status == "complete":
                status = DownloadCompleted(shard_metadata=shard, node_id=self.node_id)
                self.download_status[shard] = status
                # Footgun!
                self.event_sender.send_nowait(
                    NodeDownloadProgress(download_progress=status)
                )
                self.event_sender.send_nowait(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
            elif (
                progress.status == "in_progress"
                and current_time() - last_progress_time > throttle_interval_secs
            ):
                status = DownloadOngoing(
                    node_id=self.node_id,
                    shard_metadata=shard,
                    download_progress=map_repo_download_progress_to_download_progress_data(
                        progress
                    ),
                )
                self.download_status[shard] = status
                self.event_sender.send_nowait(
                    NodeDownloadProgress(download_progress=status)
                )
                last_progress_time = current_time()

        self.shard_downloader.on_progress(download_progress_callback)
        assert self._tg
        self._tg.start_soon(self.shard_downloader.ensure_shard, task.shard_metadata)

    async def _forward_events(self) -> None:
        with self.event_receiver as events:
            async for event in events:
                fe = ForwarderEvent(
                    origin_idx=self.local_event_index,
                    origin=self.node_id,
                    session=self.session_id,
                    event=event,
                )
                logger.debug(
                    f"Worker published event {self.local_event_index}: {str(event)[:100]}"
                )
                self.local_event_index += 1
                await self.local_event_sender.send(fe)
                self.out_for_delivery[event.event_id] = fe

    async def _poll_connection_updates(self):
        while True:
            # TODO: EdgeDeleted
            edges = set(self.state.topology.list_connections())
            conns = await connect_all(self.state.topology)
            for nid in conns:
                for ip in conns[nid]:
                    edge = Connection(
                        local_node_id=self.node_id,
                        send_back_node_id=nid,
                        send_back_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/8000")
                        if "." in ip
                        else Multiaddr(address=f"/ip6/{ip}/tcp/8000"),
                    )
                    if edge not in edges:
                        logger.debug(f"manually discovered {edge=}")
                        await self.event_sender.send(TopologyEdgeCreated(edge=edge))

            await anyio.sleep(10)


def event_relevant_to_worker(event: Event, worker: Worker):
    # TODO
    return True
