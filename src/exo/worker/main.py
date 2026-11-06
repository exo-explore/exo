import contextlib
import time
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone

import anyio
from anyio import BrokenResourceError, ClosedResourceError, to_thread
from loguru import logger

from exo.download.download_utils import is_read_only_model_dir, resolve_existing_model
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
    NodeEventReceiver,
)
from exo.routing.membership import Reachability
from exo.shared.apply import apply
from exo.shared.constants import EXO_MAX_INSTANCE_RETRIES
from exo.shared.models.model_cards import ModelCard, ModelId, card_cache
from exo.shared.request_errors import removal_message
from exo.shared.serving import why_runner_stopped
from exo.shared.types.commands import (
    AddCustomModelCard,
    ForwarderCommand,
    ForwarderDownloadCommand,
    InstanceBroken,
    StartDownload,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    EventId,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    RunnerStatusUpdated,
    StateSnapshot,
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
    LoadModel,
    Shutdown,
    Task,
    TaskStatus,
)
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadFailed
from exo.shared.types.worker.instances import InstanceId, InstanceRemovalReason
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    RunnerShuttingDown,
    RunnerStatus,
)
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from exo.utils.info_gatherer.net_profile import check_reachable
from exo.utils.keyed_backoff import KeyedBackoff
from exo.utils.task_group import TaskGroup
from exo.worker.plan import joined_to, plan
from exo.worker.runner.supervisor import RunnerSupervisor
from exo.worker.serving import ServingHost

# How often the custom model cards saved on this node are synced with the cluster state,
# and how often the ones the cluster doesn't know yet are announced to the master.
CUSTOM_CARD_SYNC_INTERVAL = 1.0
CUSTOM_CARD_ANNOUNCE_INTERVAL = 10.0
# How long the worker waits for an instance it asked to delete to go, before asking again
DELETION_REQUEST_RETRY = 10.0
# How long a runner told to shut down has to take that up itself before it is stopped
RUNNER_SHUTDOWN_GRACE = 3.0


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        *,
        event_receiver: NodeEventReceiver,
        event_sender: Sender[Event],
        # This is for requesting updates. It doesn't need to be a general command sender right now,
        # but I think it's the correct way to be thinking about commands
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        api_port: int,
        # Which nodes this node can reach now (the membership)
        reachability: Reachability,
    ):
        self.node_id: NodeId = node_id
        self.event_receiver = event_receiver
        self.event_sender = event_sender
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.api_port = api_port

        self.state: State = State()
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup = TaskGroup()

        self._system_id = SystemId()

        # Serving: requests come straight from the API nodes, not through the cluster state
        self.serving = ServingHost(
            node_id,
            state=lambda: self.state,
            runners=lambda: self.runners,
            reachability=reachability,
            api_port=api_port,
        )

        self._download_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=0.5, cap=10.0)
        # an instance asks again for its failed download at once, then after 30 s,
        # doubling up to 5 min
        self._failed_download_backoff: KeyedBackoff[InstanceId] = KeyedBackoff(
            base=15.0, cap=300.0
        )
        self._instance_backoff: KeyedBackoff[InstanceId] = KeyedBackoff(
            base=0.5, cap=10.0
        )
        # When this worker last asked for an instance it gave up on to be deleted
        self._deletion_requested: dict[InstanceId, float] = {}
        # Why instances were removed, until their runners here are stopped: what their
        # requests are told (removal_message)
        self._removal_reasons: dict[InstanceId, InstanceRemovalReason] = {}
        self._stopped: anyio.Event = anyio.Event()

        # The latest information of each kind gathered about this node, so that a new
        # master can be told all of it: some of it is only gathered once
        self._gathered: dict[type[GatheredInfo], GatheredInfo] = {}
        # After a change of master, the session whose state this worker waits for before
        # acting on the cluster state again
        self._awaiting_state_of: SessionId | None = None
        # The session of the last state snapshot applied
        self._snapshot_session: SessionId | None = None
        # Cleared once the node announces it is leaving: what it reports about itself after
        # that (its info, its connections, a model it found on disk) would add it back to the
        # cluster's state
        self._reporting = True
        # This node's runners whose process has exited, and that are yet to be replaced, with
        # what they last were (ShuttingDown or Failed): a new master may lack it
        self._gone: dict[RunnerId, RunnerStatus] = {}
        # Of those, the ones whose last report has yet to come back in this node's copy of the
        # state, by the report's event id. Until it does, that copy still holds what the old
        # process last said (Idle, Connecting...), which would be read as its replacement's.
        self._gone_unconfirmed: dict[RunnerId, EventId] = {}

    async def run(self):
        logger.info("Starting Worker")

        info_send, info_recv = channel[GatheredInfo]()
        info_gatherer: InfoGatherer = InfoGatherer(info_send)

        try:
            async with self._tg as tg:
                tg.start_soon(info_gatherer.run)
                tg.start_soon(self.serving.run)
                tg.start_soon(self._forward_info, info_recv)
                tg.start_soon(self.plan_step)
                tg.start_soon(self._event_applier)
                tg.start_soon(self._poll_connection_updates)
                tg.start_soon(self._reconcile_custom_cards)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            # Actual shutdown code - waits for all tasks to complete before executing.
            logger.info("Stopping Worker")
            self.event_sender.close()
            self.command_sender.close()
            self.download_command_sender.close()
            for runner in list(self.runners.values()):
                runner.shutdown()
            self._stopped.set()

    async def _forward_info(self, recv: Receiver[GatheredInfo]):
        with recv as info_stream:
            async for info in info_stream:
                self._gathered[type(info)] = info
                await self._send_info(info)

    def stop_reporting(self) -> None:
        """Stop telling the cluster about this node, which is leaving it."""
        self._reporting = False

    async def _report(
        self, event: NodeGatheredInfo | TopologyEdgeCreated | NodeDownloadProgress
    ) -> None:
        """Send an event about this node, unless it is leaving the cluster."""
        if self._reporting:
            await self.event_sender.send(event)

    async def _send_info(self, info: GatheredInfo) -> None:
        await self._report(
            NodeGatheredInfo(
                node_id=self.node_id,
                when=str(datetime.now(tz=timezone.utc)),
                info=info,
            )
        )

    async def _event_applier(self):
        with self.event_receiver as events:
            async for update in events:
                if isinstance(update, StateSnapshot):
                    self.state = update.state
                    self._snapshot_session = update.session
                    # A report folded into the snapshot never arrives on its own
                    for runner_id in list(self._gone_unconfirmed):
                        if isinstance(
                            self.state.runners.get(runner_id),
                            (RunnerFailed, RunnerShuttingDown),
                        ):
                            del self._gone_unconfirmed[runner_id]
                    # ... nor does an instance's deletion
                    self._forget_gone_of_deleted_instances()
                    if update.session == self._awaiting_state_of:
                        self._awaiting_state_of = None
                    continue
                # 2. for each event, apply it to the state
                self.state = apply(self.state, event=update)
                event = update.event

                if isinstance(event, InstanceDeleted):
                    self._instance_backoff.reset(event.instance_id)
                    self._deletion_requested.pop(event.instance_id, None)
                    if any(
                        runner.bound_instance.instance.instance_id == event.instance_id
                        for runner in self.runners.values()
                    ):
                        self._removal_reasons[event.instance_id] = event.reason
                    self.serving.end_instance(
                        event.instance_id, removal_message(event.reason)
                    )
                    self._forget_gone_of_deleted_instances()

                if isinstance(event, RunnerStatusUpdated):
                    if self._gone_unconfirmed.get(event.runner_id) == event.event_id:
                        del self._gone_unconfirmed[event.runner_id]
                    if isinstance(
                        event.runner_status, (RunnerFailed, RunnerShuttingDown)
                    ):
                        for runner in self.runners.values():
                            if joined_to(runner, event.runner_id):
                                runner.peer_went = True

                # Our runner for an instance has started: the attempts before it no longer
                # count towards giving up on the instance, which is for one that can't start
                if isinstance(event, RunnerStatusUpdated) and isinstance(
                    event.runner_status, RunnerReady
                ):
                    for instance in self.state.instances.values():
                        node_to_runner = instance.shard_assignments.node_to_runner
                        if node_to_runner.get(self.node_id) == event.runner_id:
                            self._instance_backoff.reset(instance.instance_id)

    def _forget_gone_of_deleted_instances(self) -> None:
        """A gone runner of an instance that no longer exists is never replaced"""
        for runner_id in list(self._gone):
            if not any(
                runner_id in instance.shard_assignments.runner_to_shard
                for instance in self.state.instances.values()
            ):
                self._gone.pop(runner_id)
                self._gone_unconfirmed.pop(runner_id, None)

    async def _reconcile_custom_cards(self) -> None:
        """Keep the custom model cards saved on this node in sync with the cluster.

        The cluster state starts empty whenever a new master takes over (for example after
        the whole cluster restarts), so the cards saved on the nodes are the only lasting
        copy. Cards this node has saved are announced until the cluster knows them, and a
        saved card is only deleted after it was in the state and then removed from it: a
        state that is empty or still catching up never deletes anything.
        """
        unannounced = {card.model_id: card for card in await card_cache.list_custom()}
        synced: Mapping[ModelId, ModelCard] = {}
        next_announcement = 0.0
        while True:
            await anyio.sleep(CUSTOM_CARD_SYNC_INTERVAL)
            target = self.state.custom_model_cards
            for model_id, card in target.items():
                unannounced.pop(model_id, None)
                if synced.get(model_id) != card:
                    await card_cache.save(card)
            for model_id in synced.keys() - target.keys():
                await card_cache.pop(model_id)
            synced = target

            if unannounced and anyio.current_time() >= next_announcement:
                next_announcement = anyio.current_time() + CUSTOM_CARD_ANNOUNCE_INTERVAL
                for card in unannounced.values():
                    await self.command_sender.send(
                        ForwarderCommand(
                            origin=self._system_id,
                            command=AddCustomModelCard(model_card=card),
                        )
                    )

    async def plan_step(self):
        while True:
            await anyio.sleep(0.1)
            if self._awaiting_state_of is not None:
                continue
            task: Task | None = plan(
                self.node_id,
                self.runners,
                self.state.downloads,
                self.state.instances,
                self.state.runners,
                self._instance_backoff,
                self._download_backoff,
                self._failed_download_backoff,
                self._gone_unconfirmed.keys(),
            )
            if task is None:
                continue

            if isinstance(task, CreateRunner):
                iid = task.instance_id
                if self._instance_backoff.attempts(iid) >= EXO_MAX_INSTANCE_RETRIES:
                    # Plan comes back to this instance every 0.1 s until it is deleted: ask
                    # once, and again only if it is still here after a while (the request
                    # may have been lost, e.g. while the master changed)
                    requested_at = self._deletion_requested.get(iid)
                    if (
                        requested_at is None
                        or time.monotonic() - requested_at >= DELETION_REQUEST_RETRY
                    ):
                        # What the user is shown: the runner's own error, when it had one
                        last = self.state.runners.get(
                            task.bound_instance.bound_runner_id
                        )
                        message = (
                            last.error_message
                            if isinstance(last, RunnerFailed) and last.error_message
                            else f"its runner failed to start {EXO_MAX_INSTANCE_RETRIES} times"
                        )
                        logger.warning(
                            f"Instance {iid}: {message}; asking for it to be removed"
                        )
                        self._deletion_requested[iid] = time.monotonic()
                        await self.command_sender.send(
                            ForwarderCommand(
                                origin=self._system_id,
                                command=InstanceBroken(
                                    instance_id=iid,
                                    reason="StartFailed",
                                    message=message,
                                ),
                            )
                        )
                    continue

            logger.info(f"Worker plan: {task.__class__.__name__}")
            assert task.task_status
            if task.task_id not in self.state.tasks:
                await self.event_sender.send(
                    TaskCreated(task_id=task.task_id, task=task)
                )

            # lets not kill the worker if a runner is unresponsive
            match task:
                case CreateRunner():
                    self._gone.pop(task.bound_instance.bound_runner_id, None)
                    await self._create_supervisor(task)
                    self._instance_backoff.record_attempt(task.instance_id)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case DownloadModel(shard_metadata=shard, instance_id=instance_id):
                    model_id = shard.model_card.model_id
                    self._download_backoff.record_attempt(model_id)
                    if any(
                        isinstance(download, DownloadFailed)
                        and download.shard_metadata.model_card.model_id == model_id
                        for download in self.state.downloads.get(self.node_id, [])
                    ):
                        self._failed_download_backoff.record_attempt(instance_id)

                    found_path = await to_thread.run_sync(
                        resolve_existing_model, model_id, shard.model_card
                    )
                    if found_path is not None:
                        logger.info(f"Model {model_id} found at {found_path}")
                        await self._report(
                            NodeDownloadProgress(
                                download_progress=DownloadCompleted(
                                    node_id=self.node_id,
                                    shard_metadata=shard,
                                    model_directory=str(found_path),
                                    total=shard.model_card.storage_size,
                                    read_only=is_read_only_model_dir(found_path),
                                )
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Complete,
                            )
                        )
                    else:
                        await self.download_command_sender.send(
                            ForwarderDownloadCommand(
                                origin=self._system_id,
                                command=StartDownload(
                                    target_node_id=self.node_id,
                                    shard_metadata=shard,
                                ),
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Running,
                            )
                        )
                case Shutdown(runner_id=runner_id):
                    # It stays in self.runners until its process has exited: only then is
                    # it replaced (see _supervise)
                    if (runner := self.runners.get(runner_id)) is not None:
                        instance = runner.bound_instance.instance
                        await runner.stop(
                            task,
                            RUNNER_SHUTDOWN_GRACE,
                            why_runner_stopped(
                                self.state,
                                instance,
                                self._removal_reasons.pop(instance.instance_id, None),
                            ),
                        )
                    # The runner is stopped, which is what the task is for, and only the worker
                    # can say so: a runner that has died never gets the task, and one that takes
                    # it up is no longer listened to once it has acknowledged it, so its own
                    # report never arrives. Unfinished, the task would stay in the state for good.
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case LoadModel(instance_id=instance_id):
                    if (instance := self.state.instances.get(instance_id)) is not None:
                        model_id = instance.shard_assignments.model_id
                        self._download_backoff.reset(model_id)
                    self._failed_download_backoff.reset(instance_id)

                    await self._start_runner_task(task)
                case task:
                    await self._start_runner_task(task)

    def follow_new_master(self, session_id: SessionId) -> None:
        """Carry on under a new master, keeping this node's runners.

        Until the new master's state arrives, the worker doesn't act on the state it has:
        that is the old master's, which the new master's may not match. Requests carry on:
        they don't go through the master. Call this as the node switches to the new master's
        session, before its state can arrive.
        """
        if self._snapshot_session != session_id:
            self._awaiting_state_of = session_id

    async def announce_to_master(self) -> None:
        """Tell a new master everything this worker knows about its node. The master's state
        may lack it: the node may have just joined the master's cluster, and the last
        updates it sent the old master may never have arrived."""
        for info in list(self._gathered.values()):
            await self._send_info(info)
        for runner in list(self.runners.values()):
            await self.event_sender.send(
                RunnerStatusUpdated(
                    runner_id=runner.bound_instance.bound_runner_id,
                    runner_status=runner.status,
                )
            )
        # Runners that are gone too: their instance's other runners must still be stopped,
        # and this node waits for the report to come back before it replaces them
        for runner_id in list(self._gone):
            # The plan may replace one during a send: its replacement isn't going
            if runner_id in self._gone and runner_id not in self.runners:
                await self._report_gone(runner_id, self._gone[runner_id])

    async def _report_gone(self, runner_id: RunnerId, status: RunnerStatus) -> None:
        event = RunnerStatusUpdated(runner_id=runner_id, runner_status=status)
        self._gone[runner_id] = status
        self._gone_unconfirmed[runner_id] = event.event_id
        with contextlib.suppress(ClosedResourceError, BrokenResourceError):
            await self.event_sender.send(event)

    async def shutdown(self):
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _start_runner_task(self, task: Task):
        if (instance := self.state.instances.get(task.instance_id)) is None:
            return
        runner_id = instance.shard_assignments.node_to_runner[self.node_id]
        # A runner can leave the table while this task was being announced
        if (runner := self.runners.get(runner_id)) is not None:
            await runner.start_task(task)

    async def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        runner = await RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
        self._tg.start_soon(self._supervise, runner)
        return runner

    async def _supervise(self, runner: RunnerSupervisor) -> None:
        """Run the supervisor, and forget the runner once it has finished, which is once the
        runner's process has exited. Until then the runner keeps its place in self.runners,
        so nothing replaces it: a replacement started any earlier would meet the old process
        still holding the instance's ports, or pair with the instance's other runners in its
        place."""
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(runner.report_if_unstoppable)
                await runner.run()
                tg.cancel_scope.cancel()
        finally:
            runner_id = runner.bound_instance.bound_runner_id
            if self.runners.get(runner_id) is runner:
                # Its last word, sent after everything its process said (so, once it is back
                # in this node's state, nothing the old process said can follow): it is going
                with anyio.CancelScope(shield=True):
                    await self._report_gone(
                        runner_id,
                        runner.status
                        if isinstance(runner.status, (RunnerFailed, RunnerShuttingDown))
                        else RunnerShuttingDown(),
                    )
                del self.runners[runner_id]

    async def _poll_connection_updates(self):
        while True:
            edges = set(
                conn.edge for conn in self.state.topology.out_edges(self.node_id)
            )
            conns: defaultdict[NodeId, set[str]] = defaultdict(set)
            async for ip, nid in check_reachable(
                self.state.topology,
                self.node_id,
                self.state.node_network,
                api_port=self.api_port,
            ):
                if ip in conns[nid]:
                    continue
                conns[nid].add(ip)
                edge = SocketConnection(
                    # nonsense multiaddr
                    sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/{self.api_port}")
                    if "." in ip
                    # nonsense multiaddr
                    else Multiaddr(address=f"/ip6/{ip}/tcp/{self.api_port}"),
                )
                if edge not in edges:
                    logger.debug(f"ping discovered {edge=}")
                    await self._report(
                        TopologyEdgeCreated(
                            conn=Connection(source=self.node_id, sink=nid, edge=edge)
                        )
                    )

            for conn in self.state.topology.out_edges(self.node_id):
                if not isinstance(conn.edge, SocketConnection):
                    continue
                # ignore mDNS discovered connections
                if conn.edge.sink_multiaddr.port != self.api_port:
                    continue
                if (
                    conn.sink not in conns
                    or conn.edge.sink_multiaddr.ip_address not in conns[conn.sink]
                ):
                    logger.debug(f"ping failed to discover {conn=}")
                    await self.event_sender.send(TopologyEdgeDeleted(conn=conn))

            await anyio.sleep(10)
