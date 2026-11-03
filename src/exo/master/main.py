import copy
from collections import deque
from collections.abc import Set as AbstractSet
from datetime import datetime, timezone
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
from exo.master.placement_utils import find_ip_prioritised
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.routing.membership import MembershipReader, MembershipView, listed_nodes
from exo.shared.apply import apply
from exo.shared.constants import EXO_TRACING_ENABLED
from exo.shared.model_readiness import instance_is_ready, instance_readiness
from exo.shared.models.model_cards import ModelId
from exo.shared.request_errors import INSTANCE_GONE_BEFORE_START
from exo.shared.topology import Topology
from exo.shared.types.chunks import ErrorChunk, ImageChunk
from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    DeleteInstanceLink,
    DownloadCommand,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    RequestEventLog,
    SendInputChunk,
    SetInstanceLink,
    TaskCancelled,
    TaskFinished,
    TestCommand,
    TextGeneration,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InputChunkReceived,
    InstanceDeleted,
    InstanceLinkCreated,
    InstanceLinkDeleted,
    LocalForwarderEvent,
    NodeGatheredInfo,
    NodeTimedOut,
    StateSnapshot,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
    TracesMerged,
)
from exo.shared.types.instance_link import InstanceLink
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import Receiver, Sender
from exo.utils.event_buffer import MultiSourceBuffer
from exo.utils.task_group import TaskGroup

# How often the master looks for nodes to remove and for instances that lost a node. Which
# nodes are gone, and since when, is the membership's to say (rust/networking/src/
# membership.rs owns every liveness timer); this only bounds how late the master acts on it.
PLAN_INTERVAL = 1.0
# Recent events kept for nodes that missed a few. A node further behind than this
# is sent a snapshot of the current state instead of the whole history.
REPLAYABLE_EVENTS = 10_000
# ...holding at most this much image data. Image chunks are the only large events, and a
# few requests with large images would otherwise make the window hold gigabytes.
REPLAYABLE_IMAGE_BYTES = 256 * 1024 * 1024
# Events replayed per RequestEventLog; the requester asks again for the rest.
REPLAY_BATCH_SIZE = 1000


def _image_bytes(event: Event) -> int:
    match event:
        case InputChunkReceived(chunk=chunk):
            return len(chunk.data)
        case ChunkGenerated(chunk=ImageChunk() as chunk):
            return len(chunk.data)
        case _:
            return 0


def task_for_command(state: State, command_id: CommandId) -> TaskId | None:
    """The generation task created for a command, as state has it. The master's own tasks are
    in its state as soon as it creates them, and state carries over to the next master, so a
    request started under a previous master can still be finished or cancelled."""
    for task_id, task in state.tasks.items():
        if (
            isinstance(task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask))
            and task.command_id == command_id
        ):
            return task_id
    return None


def _prefill_endpoint_for(state: State, decode_instance_id: InstanceId) -> str | None:
    decode = state.instances.get(decode_instance_id)
    if decode is None:
        return None
    decode_node = next(iter(decode.shard_assignments.node_to_runner.keys()), None)
    if decode_node is None:
        return None

    sources: set[InstanceId] = set()
    for link in state.instance_links.values():
        if decode_instance_id in link.decode_instances:
            sources.update(link.prefill_instances)
    sources.discard(decode_instance_id)

    in_flight = {TaskStatus.Pending, TaskStatus.Running}
    task_counts: dict[InstanceId, int] = {
        src_id: sum(
            1
            for task in state.tasks.values()
            if task.instance_id == src_id and task.task_status in in_flight
        )
        for src_id in sources
    }
    for src_id in sorted(sources, key=lambda sid: task_counts[sid]):
        instance = state.instances.get(src_id)
        if instance is None:
            continue
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            port = state.prefill_server_ports.get(runner_id)
            if port is None:
                continue
            ip = find_ip_prioritised(
                decode_node, node_id, state.topology, state.node_network, ring=True
            )
            if ip is None:
                continue
            return f"{ip}:{port}"
    return None


def _without_requests(state: State) -> State:
    """The state without its requests' tasks. Requests don't outlast their master: every
    node's API ends the ones in flight when the master changes, since the old master was
    routing their output, so their tasks would only linger."""
    return state.model_copy(
        update={
            "tasks": {
                task_id: task
                for task_id, task in state.tasks.items()
                if not isinstance(
                    task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask)
                )
            }
        }
    )


# How many processed command ids the master remembers, to act on a repeated command only once
PROCESSED_COMMANDS_KEPT = 100_000


def gone_nodes(listed: AbstractSet[NodeId], view: MembershipView) -> set[NodeId]:
    """The listed nodes Gone in this node's membership view (Suspect for the grace period, or
    listed and never seen for as long), as far as its quorum read lets it act on them: their
    instances can't serve, so they are ended. Never this node itself.

    - Majority, or Half (two nodes with one away, or an even split): all of them.
    - Minority: none. This node is most likely the one cut off, and the rest of the cluster
      still runs what it can't see.

    Suspect nodes are never among them: they may be back within the grace period. Their
    requests end at once anyway (every API checks reachability itself).
    """
    if view.quorum == "Minority":
        return set()
    return {
        member.node_id
        for member in view.members
        if member.state == "Gone"
        and member.node_id in listed
        and member.node_id != view.node_id
    }


def nodes_to_remove(gone: AbstractSet[NodeId], view: MembershipView) -> set[NodeId]:
    """Which of the gone nodes the coordinator removes from the cluster now (D4):

    - Majority: all of them.
    - Half: none yet. The node stays listed as Gone, and its instances end; it is removed
      once it has been Gone for FORGET_AFTER (the split hold, SPLIT_HOLD): it then stops
      counting, and the read is Majority. Removing it at once would make this side a
      majority of one, which must not happen before the split hold. Once deployments
      exist (step 3), this is also where the other side's deployments are kept, and
      re-placed only after SPLIT_HOLD unless the user acts.
    - Minority: none (no node is gone, see gone_nodes). Once they have been Gone for
      FORGET_AFTER, they stop counting in its quorum read, which then is a majority of the
      nodes it reaches, and they are removed.
    """
    match view.quorum:
        case "Majority":
            return set(gone)
        case "Half" | "Minority":
            return set()


class NoInstanceForModelError(Exception):
    """A request for a model with no instance that can serve it: none at all (one deleted after
    the API checked for it), or none ready. The request is ended with this error: dropping it
    would leave it open forever, and sending it to an instance that isn't ready would leave it
    waiting for as long as the instance takes, perhaps forever."""

    def __init__(self, model: str, message: str):
        super().__init__(message)
        self.model = ModelId(model)


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
        membership: MembershipReader,
        initial_state: State | None = None,
    ):
        """global_event_sender must never block (the router's channels are unbounded): the
        master sends each event as it indexes it, without awaiting."""
        self.node_id = node_id
        # This node's view of which nodes are reachable and gone, and of how much of the
        # cluster it reaches: what removals and the choice of instance go by
        self.membership = membership
        self.session_id = session_id
        # A master elected in a running cluster carries on from the state its node already
        # has, so model instances survive the change of master. Its events are numbered on
        # from that state, and other nodes catch up from a snapshot of it.
        self.state = (
            _without_requests(initial_state) if initial_state is not None else State()
        )
        self._tg: TaskGroup = TaskGroup()
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.snapshot_sender = snapshot_sender
        self.download_command_sender = download_command_sender
        self._system_id = SystemId()
        self._multi_buffer = MultiSourceBuffer[SystemId, Event]()
        self._recent_events: deque[Event] = deque()
        self._recent_image_bytes = 0
        self._replayable_events = REPLAYABLE_EVENTS
        self._replayable_image_bytes = REPLAYABLE_IMAGE_BYTES
        self._pending_traces: dict[TaskId, dict[int, list[TraceEventData]]] = {}
        # Set when a node is removed, so its instances are deleted at once rather than at the
        # next check (created by the plan loop, which waits on it)
        self._node_removed: anyio.Event | None = None
        # The gone nodes kept listed because this node's quorum read isn't Majority
        self._held_back: frozenset[NodeId] = frozenset()
        self._processed_commands: set[CommandId] = set()
        self._processed_order: deque[CommandId] = deque()
        self._expected_ranks: dict[TaskId, set[int]] = {}

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
        """Whether this command was processed already. The API sends a request again until it
        sees it take effect, so the same command can arrive more than once, even after what it
        did was undone (a launch stopped since): a copy is ignored. A new master doesn't know
        what the old one processed, but a request it accepted is a task."""
        return command_id in self._processed_commands or self._has_task_for(command_id)

    def _remember_processed(self, command_id: CommandId) -> None:
        """Only a command that changed something is remembered: one that changed nothing (a
        launch onto a node outside the cluster, say) may take effect when a copy comes."""
        self._processed_commands.add(command_id)
        self._processed_order.append(command_id)
        if len(self._processed_order) > PROCESSED_COMMANDS_KEPT:
            self._processed_commands.discard(self._processed_order.popleft())

    def _has_task_for(self, command_id: CommandId) -> bool:
        return any(
            isinstance(task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask))
            and task.command_id == command_id
            for task in self.state.tasks.values()
        )

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
                        case TextGeneration():
                            # set-difference => prefill-only nodes
                            prefill_only: set[InstanceId] = set()
                            for link in self.state.instance_links.values():
                                prefill_only.update(link.prefill_instances)
                            for link in self.state.instance_links.values():
                                prefill_only.difference_update(link.decode_instances)

                            decode_instance_id = self._least_busy_instance(
                                command.task_params.model, excluded=prefill_only
                            )
                            task_id = TaskId()
                            params = command.task_params.model_copy(
                                update={
                                    "prefill_endpoint": _prefill_endpoint_for(
                                        self.state, decode_instance_id
                                    ),
                                }
                            )
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=TextGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=decode_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=params,
                                    ),
                                )
                            )
                        case ImageGeneration():
                            task_id = TaskId()
                            selected_instance_id = self._least_busy_instance(
                                command.task_params.model
                            )
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case ImageEdits():
                            task_id = TaskId()
                            selected_instance_id = self._least_busy_instance(
                                command.task_params.model
                            )
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageEditsTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
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
                                self.state.instances, placement, self.state.tasks
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
                                self._reachable_topology(),
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
                                self.state.instances, placement, self.state.tasks
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
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case SendInputChunk(chunk=chunk):
                            generated_events.append(
                                InputChunkReceived(
                                    command_id=chunk.command_id,
                                    chunk=chunk,
                                )
                            )
                        case TaskCancelled():
                            if (
                                task_id := task_for_command(
                                    self.state, command.cancelled_command_id
                                )
                            ) is not None:
                                generated_events.append(
                                    TaskStatusUpdated(
                                        task_status=TaskStatus.Cancelled,
                                        task_id=task_id,
                                    )
                                )
                            else:
                                logger.warning(
                                    f"Nonexistent command {command.cancelled_command_id} cancelled"
                                )
                        case TaskFinished():
                            # The API resends a finish while the task is still in its state,
                            # so a finish for a task already deleted is expected
                            if (
                                task_id := task_for_command(
                                    self.state, command.finished_command_id
                                )
                            ) is not None:
                                generated_events.append(TaskDeleted(task_id=task_id))

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
                except NoInstanceForModelError as error:
                    command_id = forwarder_command.command.command_id
                    self._remember_processed(command_id)
                    logger.warning(f"Ending request {command_id}: {error}")
                    self._index(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=ErrorChunk(
                                model=error.model, error_message=str(error)
                            ),
                        )
                    )
                except Exception as e:
                    logger.opt(exception=e).warning("Error in command processor")

    def _least_busy_instance(
        self, model: str, excluded: AbstractSet[InstanceId] = frozenset()
    ) -> InstanceId:
        """The ready instance of the model with the fewest requests in flight.

        Instances that aren't ready (downloading, loading, failed) are left out: the API waits
        for a model to be ready before sending a request, but with several instances of a
        model, one may be ready and another not.
        """
        in_flight = {TaskStatus.Pending, TaskStatus.Running}
        of_model = [
            instance
            for instance_id, instance in self.state.instances.items()
            if instance.shard_assignments.model_id == model
            and instance_id not in excluded
        ]
        if not of_model:
            # The API sends a request only for a model it saw an instance of, so the instance
            # was deleted (the model stopped, or a node lost) after the API looked
            raise NoInstanceForModelError(model, INSTANCE_GONE_BEFORE_START)
        ready = [
            instance.instance_id
            for instance in of_model
            if instance_is_ready(instance, self.state.runners)
        ]
        if not ready:
            not_ready = [
                status
                for instance in of_model
                if (status := instance_readiness(self.state, instance)) is not None
            ]
            raise NoInstanceForModelError(
                model,
                min(
                    not_ready,
                    key=lambda status: (status.failed, status.retry_after_seconds),
                ).message,
            )
        task_counts: dict[InstanceId, int] = {
            instance_id: sum(
                1
                for task in self.state.tasks.values()
                if task.instance_id == instance_id and task.task_status in in_flight
            )
            for instance_id in ready
        }
        return min(task_counts, key=lambda instance_id: task_counts[instance_id])

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            self._node_removed = anyio.Event()
            self._remove_silent_nodes_and_broken_instances()
            with anyio.move_on_after(PLAN_INTERVAL):
                await self._node_removed.wait()

    def _remove_silent_nodes_and_broken_instances(self) -> None:
        # Gone in this node's membership view, as far as its quorum read allows: a node that
        # was stopped itself (frozen, or its Mac asleep) is Resuming for a while, and doesn't
        # time the peers it lost meanwhile toward Gone, so it doesn't blame them for its stall
        view = self.membership.view()
        listed = listed_nodes(self.state)
        gone = gone_nodes(listed, view)
        self._note_held_back(listed, view)
        members = {member.node_id: member for member in view.members}
        # Each removal is in the state as soon as it is made, so a node or an instance is
        # removed once
        for node_id in nodes_to_remove(gone, view):
            member = members[node_id]
            logger.info(
                f"Removing node {node_id}: gone ({member.reason}) since "
                f"{member.since.isoformat(timespec='seconds')}"
            )
            self._index(NodeTimedOut(node_id=node_id))

        # An instance that lost a node can't serve anything, so delete it straight away (in
        # an even split, while the node stays listed)
        connected = set(self.state.topology.list_nodes()) - gone
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

    def _reachable_topology(self) -> Topology:
        """The topology without the nodes this node can't reach now, as its membership sees
        them: a model isn't placed on a node that can't serve it (one Suspect, or one Gone
        that an even split keeps listed)"""
        unreachable = [
            node
            for node in self.state.topology.list_nodes()
            if not self.membership.is_reachable(node)
        ]
        if not unreachable:
            return self.state.topology
        topology = copy.deepcopy(self.state.topology)
        for node in unreachable:
            topology.remove_node(node)
        return topology

    def _note_held_back(
        self, listed: AbstractSet[NodeId], view: MembershipView
    ) -> None:
        """Logs when the gone nodes kept listed because of this node's quorum read change"""
        held_back = frozenset(
            member.node_id
            for member in view.members
            if view.quorum != "Majority"
            and member.state == "Gone"
            and member.node_id in listed
            and member.node_id != view.node_id
        )
        if held_back and held_back != self._held_back:
            if view.quorum == "Minority":
                logger.warning(
                    f"Not removing {len(held_back)} gone node(s) or their instances: this "
                    f"node reaches {view.reachable} of the {view.last_known} nodes it knows, "
                    "so it is most likely the one cut off"
                )
            else:
                logger.warning(
                    f"Keeping {len(held_back)} gone node(s) listed, and ending their "
                    f"instances: this node reaches {view.reachable} of the "
                    f"{view.last_known} nodes it knows, exactly half, so it removes them "
                    "only after the split hold"
                )
        elif self._held_back and not held_back:
            logger.info("No longer holding back the removal of gone nodes")
        self._held_back = held_back

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
                    if isinstance(event, TracesCollected):
                        self._handle_traces_collected(event)
                        continue

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
        self._recent_image_bytes += _image_bytes(event)
        while (
            len(self._recent_events) > self._replayable_events
            or self._recent_image_bytes > self._replayable_image_bytes
        ):
            self._recent_image_bytes -= _image_bytes(self._recent_events.popleft())

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

    def _handle_traces_collected(self, event: TracesCollected) -> None:
        task_id = event.task_id
        if task_id not in self._pending_traces:
            self._pending_traces[task_id] = {}
        self._pending_traces[task_id][event.rank] = event.traces

        if (
            task_id in self._expected_ranks
            and set(self._pending_traces[task_id].keys())
            >= self._expected_ranks[task_id]
        ):
            self._merge_and_save_traces(task_id)

    def _merge_and_save_traces(self, task_id: TaskId) -> None:
        all_trace_data: list[TraceEventData] = []
        for trace_data in self._pending_traces[task_id].values():
            all_trace_data.extend(trace_data)

        self._index(TracesMerged(task_id=task_id, traces=all_trace_data))

        del self._pending_traces[task_id]
        if task_id in self._expected_ranks:
            del self._expected_ranks[task_id]
