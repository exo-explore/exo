from datetime import datetime, timezone
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
    NodeGatheredInfo,
    TaskCreated,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CreateRunner,
    DownloadModel,
    Shutdown,
    Task,
    TaskStatus,
)
from exo.shared.types.topology import SocketConnection
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
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from exo.utils.info_gatherer.net_profile import check_reachable
from exo.worker.download.download_utils import (
    map_repo_download_progress_to_download_progress_data,
)
from exo.worker.download.shard_downloader import RepoDownloadProgress, ShardDownloader
from exo.worker.plan import plan
from exo.worker.runner.runner_supervisor import RunnerSupervisor


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        shard_downloader: ShardDownloader,
        *,
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
        self.out_for_delivery: dict[EventId, ForwarderEvent] = {}

        self.state: State = State()
        self.download_status: dict[ShardMetadata, DownloadProgress] = {}
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup = create_task_group()

        self._nack_cancel_scope: CancelScope | None = None
        self._nack_attempts: int = 0
        self._nack_base_seconds: float = 0.5
        self._nack_cap_seconds: float = 10.0

        self.event_sender, self.event_receiver = channel[Event]()

    async def run(self):
        logger.info("Starting Worker")

        info_send, info_recv = channel[GatheredInfo]()
        info_gatherer: InfoGatherer = InfoGatherer(info_send)

        async with self._tg as tg:
            tg.start_soon(info_gatherer.run)
            tg.start_soon(self._forward_info, info_recv)
            tg.start_soon(self.plan_step)
            tg.start_soon(self._connection_message_event_writer)
            tg.start_soon(self._resend_out_for_delivery)
            tg.start_soon(self._event_applier)
            tg.start_soon(self._forward_events)
            tg.start_soon(self._poll_connection_updates)

        # Actual shutdown code - waits for all tasks to complete before executing.
        self.local_event_sender.close()
        self.command_sender.close()
        for runner in self.runners.values():
            runner.shutdown()

    async def _forward_info(self, recv: Receiver[GatheredInfo]):
        with recv as info_stream:
            async for info in info_stream:
                await self.event_sender.send(
                    NodeGatheredInfo(
                        node_id=self.node_id,
                        when=str(datetime.now(tz=timezone.utc)),
                        info=info,
                    )
                )

    async def _event_applier(self):
        with self.global_event_receiver as events:
            async for f_event in events:
                if f_event.origin != self.session_id.master_node_id:
                    continue
                self.event_buffer.ingest(f_event.origin_idx, f_event.event)
                event_id = f_event.event.event_id
                if event_id in self.out_for_delivery:
                    del self.out_for_delivery[event_id]

                # 2. for each event, apply it to the state
                indexed_events = self.event_buffer.drain_indexed()
                if indexed_events:
                    self._nack_attempts = 0

                if not indexed_events and (
                    self._nack_cancel_scope is None
                    or self._nack_cancel_scope.cancel_called
                ):
                    # Request the next index.
                    self._tg.start_soon(
                        self._nack_request, self.state.last_event_applied_idx + 1
                    )
                    continue
                elif indexed_events and self._nack_cancel_scope:
                    self._nack_cancel_scope.cancel()

                for idx, event in indexed_events:
                    self.state = apply(self.state, IndexedEvent(idx=idx, event=event))

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
                    source=self.node_id,
                    sink=msg.node_id,
                    edge=SocketConnection(
                        sink_multiaddr=Multiaddr(
                            address=f"/ip4/{msg.remote_ipv4}/tcp/{msg.remote_tcp_port}"
                        ),
                    ),
                )

            case ConnectionMessageType.Disconnected:
                return TopologyEdgeDeleted(
                    source=self.node_id,
                    sink=msg.node_id,
                    edge=SocketConnection(
                        sink_multiaddr=Multiaddr(
                            address=f"/ip4/{msg.remote_ipv4}/tcp/{msg.remote_tcp_port}"
                        ),
                    ),
                )

    async def _nack_request(self, since_idx: int) -> None:
        # We request all events after (and including) the missing index.
        # This function is started whenever we receive an event that is out of sequence.
        # It is cancelled as soon as we receiver an event that is in sequence.

        if since_idx < 0:
            logger.warning(f"Negative value encountered for nack request {since_idx=}")
            since_idx = 0

        with CancelScope() as scope:
            self._nack_cancel_scope = scope
            delay: float = self._nack_base_seconds * (2.0**self._nack_attempts)
            delay = min(self._nack_cap_seconds, delay)
            self._nack_attempts += 1
            try:
                await anyio.sleep(delay)
                logger.info(
                    f"Nack attempt {self._nack_attempts}: Requesting Event Log from {since_idx}"
                )
                await self.command_sender.send(
                    ForwarderCommand(
                        origin=self.node_id,
                        command=RequestEventLog(since_idx=since_idx),
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
            conns = await check_reachable(self.state.topology, self.state.node_profiles)
            for nid in conns:
                for ip in conns[nid]:
                    edge = SocketConnection(
                        # nonsense multiaddr
                        sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415")
                        if "." in ip
                        # nonsense multiaddr
                        else Multiaddr(address=f"/ip6/{ip}/tcp/52415"),
                    )
                    if edge not in edges:
                        logger.debug(f"ping discovered {edge=}")
                        await self.event_sender.send(
                            TopologyEdgeCreated(
                                source=self.node_id, sink=nid, edge=edge
                            )
                        )

            for nid, conn in self.state.topology.out_edges(self.node_id):
                if not isinstance(conn, SocketConnection):
                    continue
                if nid not in conns or conn.sink_multiaddr.ip_address not in conns.get(
                    nid, set()
                ):
                    logger.debug(f"ping failed to discover {conn=}")
                    await self.event_sender.send(
                        TopologyEdgeDeleted(source=self.node_id, sink=nid, edge=conn)
                    )

            await anyio.sleep(10)
