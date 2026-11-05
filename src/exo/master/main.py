import time
from collections import deque
from datetime import datetime, timedelta, timezone
from itertools import islice

import anyio
from loguru import logger

from exo.master.placement import (
    InstanceDoesNotFitError,
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    get_transition_events,
    place_instance,
)
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.shared.apply import apply
from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    DeleteInstanceLink,
    DownloadCommand,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
    RequestEventLog,
    SetInstanceLink,
    TestCommand,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceDeleted,
    InstanceLinkCreated,
    InstanceLinkDeleted,
    LocalForwarderEvent,
    NodeGatheredInfo,
    NodeTimedOut,
    StateSnapshot,
)
from exo.shared.types.instance_link import InstanceLink
from exo.shared.types.state import State
from exo.utils.channels import Receiver, Sender
from exo.utils.event_buffer import MultiSourceBuffer
from exo.utils.task_group import TaskGroup

# Every node reports its info about once a second. A node the master hasn't heard from for
# this long is treated as gone (crashed, frozen, asleep or cut off), and its instances are
# deleted so that requests on them end with an error instead of waiting. On four busy nodes
# (800 events/s) a healthy node's reports were never more than 6.3 s apart.
NODE_SILENCE_TIMEOUT = timedelta(seconds=15)
# How often the master looks for silent nodes and for instances that lost a node
PLAN_INTERVAL = 1.0
# A check this late means the master itself was stalled (frozen, asleep or overloaded) and
# couldn't hear anyone. Nodes then get time to reconnect (discovery holds off for a while
# after a stall) and a full NODE_SILENCE_TIMEOUT to report again before any is removed.
MASTER_STALL = 5.0
RECONNECT_ALLOWANCE = timedelta(seconds=15)
# Recent events kept for nodes that missed a few. A node further behind than this
# is sent a snapshot of the current state instead of the whole history.
REPLAYABLE_EVENTS = 10_000
# Events replayed per RequestEventLog; the requester asks again for the rest.
REPLAY_BATCH_SIZE = 1000


# How many processed command ids the master remembers, to act on a repeated command only once
PROCESSED_COMMANDS_KEPT = 100_000


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        local_event_receiver: Receiver[LocalForwarderEvent],
        global_event_sender: Sender[GlobalForwarderEvent],
        snapshot_sender: Sender[StateSnapshot],
        download_command_sender: Sender[ForwarderDownloadCommand],
        initial_state: State | None = None,
    ):
        """global_event_sender must never block (the router's channels are unbounded): the
        master sends each event as it indexes it, without awaiting."""
        self.node_id = node_id
        self.session_id = session_id
        # A master elected in a running cluster carries on from the state its node already
        # has, so model instances survive the change of master. Its events are numbered on
        # from that state, and other nodes catch up from a snapshot of it. Requests aren't in
        # it: they don't go through the master, so they don't notice the change.
        self.state = initial_state if initial_state is not None else State()
        self._tg: TaskGroup = TaskGroup()
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.snapshot_sender = snapshot_sender
        self.download_command_sender = download_command_sender
        self._system_id = SystemId()
        self._multi_buffer = MultiSourceBuffer[SystemId, Event]()
        self._recent_events: deque[Event] = deque()
        self._replayable_events = REPLAYABLE_EVENTS
        # Since when the master has been able to hear nodes, and when it last checked on them
        self._listening_since = time.monotonic()
        self._last_check: float | None = None
        # Set when a node is removed, so its instances are deleted at once rather than at the
        # next check (created by the plan loop, which waits on it)
        self._node_removed: anyio.Event | None = None
        self._processed_commands: set[CommandId] = set()
        self._processed_order: deque[CommandId] = deque()

    async def run(self):
        logger.info("Starting Master")

        try:
            async with self._tg as tg:
                tg.start_soon(self._event_processor)
                tg.start_soon(self._command_processor)
                tg.start_soon(self._plan)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            self.global_event_sender.close()
            self.snapshot_sender.close()
            self.local_event_receiver.close()
            self.command_receiver.close()

    async def shutdown(self):
        logger.info("Stopping Master")
        self._tg.cancel_tasks()

    def _already_processed(self, command_id: CommandId) -> bool:
        """Whether this command was processed already. The API sends a change again until it
        sees it take effect, so the same command can arrive more than once, even after what it
        did was undone (a launch stopped since): a copy is ignored."""
        return command_id in self._processed_commands

    def _remember_processed(self, command_id: CommandId) -> None:
        """Only a command that changed something is remembered: one that changed nothing (a
        launch onto a node outside the cluster, say) may take effect when a copy comes."""
        self._processed_commands.add(command_id)
        self._processed_order.append(command_id)
        if len(self._processed_order) > PROCESSED_COMMANDS_KEPT:
            self._processed_commands.discard(self._processed_order.popleft())

    async def _command_processor(self) -> None:
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                if self._already_processed(forwarder_command.command.command_id):
                    logger.debug(
                        f"Ignoring command {forwarder_command.command.command_id}: already processed"
                    )
                    continue
                try:
                    logger.info(f"Executing command: {forwarder_command.command}")

                    generated_events: list[Event] = []
                    download_commands: list[DownloadCommand] = []
                    command = forwarder_command.command
                    match command:
                        case TestCommand():
                            pass
                        case DeleteInstance() if (
                            command.instance_id not in self.state.instances
                        ):
                            # A copy sent before its node saw the instance go
                            logger.debug(
                                f"Not deleting instance {command.instance_id}: already gone"
                            )
                        case DeleteInstance():
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement
                            )
                            download_commands.extend(
                                cancel_unnecessary_downloads(
                                    placement, self.state.downloads
                                )
                            )
                            generated_events.extend(transition_events)
                        case PlaceInstance():
                            placement = place_instance(
                                command,
                                self.state.topology,
                                self.state.instances,
                                self.state.node_memory,
                                self.state.node_network,
                                self.state.node_backends,
                                download_status=self.state.downloads,
                                node_rdma_ctl=self.state.node_rdma_ctl,
                                runners=self.state.runners,
                                node_identities=self.state.node_identities,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            try:
                                placement = add_instance_to_placements(
                                    command, self.state
                                )
                            except InstanceDoesNotFitError as error:
                                # Its request learns of it only as the launch not showing
                                logger.warning(
                                    f"Not launching instance "
                                    f"{command.instance.instance_id}: {error}"
                                )
                                placement = self.state.instances
                            transition_events = get_transition_events(
                                self.state.instances, placement
                            )
                            generated_events.extend(transition_events)
                        case AddCustomModelCard():
                            generated_events.append(
                                CustomModelCardAdded(model_card=command.model_card)
                            )
                        case DeleteCustomModelCard():
                            generated_events.append(
                                CustomModelCardDeleted(model_id=command.model_id)
                            )
                        case SetInstanceLink():
                            link = InstanceLink(
                                link_id=command.link_id,
                                prefill_instances=list(
                                    dict.fromkeys(command.prefill_instances)
                                ),
                                decode_instances=list(
                                    dict.fromkeys(command.decode_instances)
                                ),
                            )
                            generated_events.append(InstanceLinkCreated(link=link))
                        case DeleteInstanceLink():
                            generated_events.append(
                                InstanceLinkDeleted(link_id=command.link_id)
                            )
                        case RequestEventLog():
                            await self._serve_event_log_request(
                                command.since_idx,
                                forwarder_command.origin,
                                snapshot=command.snapshot,
                            )
                    if generated_events:
                        self._remember_processed(command.command_id)
                    # Applied before the next command is looked at, with nothing awaited since
                    # the state they were decided on was read: every decision sees the ones
                    # before it. All of a command's events apply, or none (see _index), and
                    # its download commands are sent only once they have.
                    self._index(*generated_events)
                    for download_command in download_commands:
                        await self.download_command_sender.send(
                            ForwarderDownloadCommand(
                                origin=self._system_id, command=download_command
                            )
                        )
                except Exception as e:
                    logger.opt(exception=e).warning("Error in command processor")

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            self._node_removed = anyio.Event()
            self._remove_silent_nodes_and_broken_instances()
            with anyio.move_on_after(PLAN_INTERVAL):
                await self._node_removed.wait()

    def _remove_silent_nodes_and_broken_instances(self) -> None:
        check = time.monotonic()
        if self._last_check is not None and check - self._last_check > MASTER_STALL:
            logger.warning(
                f"Master was unresponsive for {check - self._last_check:.0f}s; "
                "giving nodes time to report before removing any"
            )
            self._listening_since = check + RECONNECT_ALLOWANCE.total_seconds()
        self._last_check = check
        # A node is silent only if the master was listening and didn't hear from it
        listened = timedelta(seconds=check - self._listening_since)
        now = datetime.now(tz=timezone.utc)
        silent = {
            node_id: now - seen
            for node_id, seen in self.state.last_seen.items()
            if min(now - seen, listened) > NODE_SILENCE_TIMEOUT
        }
        # Each removal is in the state as soon as it is made, so a node or an instance is
        # removed once
        for node_id, silence in silent.items():
            logger.info(
                f"Manually removing node {node_id} due to inactivity "
                f"({silence.total_seconds():.0f}s without hearing from it)"
            )
            self._index(NodeTimedOut(node_id=node_id))

        # An instance that lost a node can't serve anything, so delete it straight away
        connected = set(self.state.topology.list_nodes())
        for instance_id, instance in list(self.state.instances.items()):
            if any(
                node_id not in connected
                for node_id in instance.shard_assignments.node_to_runner
            ):
                self._index(InstanceDeleted(instance_id=instance_id))

    def _on_node_removed(self, node_id: NodeId) -> None:
        # Only for a removal a node sent: it announced it was leaving. The ones this master
        # makes are in its state at once, and the check that made them deletes the node's
        # instances in the same pass.
        logger.info(f"Node {node_id} left the cluster")
        # Delete the instances it was part of now, not at the next check
        if self._node_removed is not None:
            self._node_removed.set()

    async def _event_processor(self) -> None:
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                # Discard all events not from our session
                if local_event.session != self.session_id:
                    continue
                self._multi_buffer.ingest(
                    local_event.origin_idx,
                    local_event.event,
                    local_event.origin,
                )
                for event in self._multi_buffer.drain():
                    logger.debug(f"Master indexing event: {str(event)[:100]}")
                    self._index(event)
                    if isinstance(event, NodeTimedOut):
                        self._on_node_removed(event.node_id)

    def _index(self, *events: Event) -> None:
        """Give events the next indices, apply them to the state, keep them for replay and send
        them to every node (this one included).

        Every event goes through here: the ones nodes send, and the master's own as soon as it
        decides on them. Nothing is awaited, so nothing else runs between choosing the indices
        and applying the events: whatever the master decides next sees them, and events go out
        in the order of their indices.

        The events of one call take effect together or not at all, so a command is never half
        applied: they are applied to the state in turn (apply never changes the state it is
        given), and only if all of them apply is the result kept and are they sent. If one
        fails, the error is raised and the state is as it was.
        """
        state = self.state
        indexed_events: list[IndexedEvent] = []
        for event in events:
            event = event.model_copy(
                update={"_master_time_stamp": datetime.now(tz=timezone.utc)}
            )
            if isinstance(event, NodeGatheredInfo):
                event = event.model_copy(
                    update={"when": str(datetime.now(tz=timezone.utc))}
                )
            indexed = IndexedEvent(event=event, idx=state.last_event_applied_idx + 1)
            state = apply(state, indexed)
            indexed_events.append(indexed)

        self.state = state
        for indexed in indexed_events:
            self._remember(indexed.event)
            self._send_indexed_event(indexed)

    def _remember(self, event: Event) -> None:
        self._recent_events.append(event)
        while len(self._recent_events) > self._replayable_events:
            self._recent_events.popleft()

    async def _serve_event_log_request(
        self, since_idx: int, requester: SystemId, snapshot: bool = False
    ):
        next_idx = self.state.last_event_applied_idx + 1
        oldest_idx = next_idx - len(self._recent_events)
        if snapshot or since_idx < oldest_idx:
            logger.info(
                f"Sending a state snapshot at event {next_idx - 1} to a node "
                f"that asked for events from {since_idx}"
            )
            await self.snapshot_sender.send(
                StateSnapshot(
                    session=self.session_id, requester=requester, state=self.state
                )
            )
            return
        end = min(since_idx + REPLAY_BATCH_SIZE, next_idx)
        events = list(
            islice(self._recent_events, since_idx - oldest_idx, end - oldest_idx)
        )
        # Other nodes ignore events they already have
        for idx, event in enumerate(events, start=since_idx):
            self._send_indexed_event(IndexedEvent(idx=idx, event=event))

    def _send_indexed_event(self, event: IndexedEvent) -> None:
        self.global_event_sender.send_nowait(
            GlobalForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )
