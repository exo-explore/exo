from collections.abc import Sequence
from datetime import datetime, timezone

import anyio
from anyio.abc import TaskGroup
from loguru import logger

from exo.master.event_log import DiskEventLog
from exo.master.placement import (
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    get_transition_events,
    place_instance,
)
from exo.master.process_managers import ProcessManager
from exo.master.process_managers.instance_health import InstanceHealthReconciler
from exo.master.process_managers.meta_instance import MetaInstanceReconciler
from exo.master.process_managers.node_timeout import NodeTimeoutReconciler
from exo.master.reconcile import (
    find_unsatisfied_meta_instances,
    try_place_for_meta_instance,
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_EVENT_LOG_DIR, EXO_TRACING_ENABLED
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.commands import (
    CreateInstance,
    CreateMetaInstance,
    DeleteInstance,
    DeleteMetaInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    RequestEventLog,
    SendInputChunk,
    TaskCancelled,
    TaskFinished,
    TestCommand,
    TextGeneration,
)
from exo.shared.types.common import CommandId, NodeId, SessionId
from exo.shared.types.events import (
    Event,
    ForwarderEvent,
    IndexedEvent,
    InputChunkReceived,
    InstanceDeleted,
    JacclSideChannelData,
    JacclSideChannelGathered,
    MetaInstanceCreated,
    MetaInstanceDeleted,
    MetaInstancePlacementFailed,
    NodeGatheredInfo,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
    TracesMerged,
)
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
from exo.shared.types.worker.runners import RunnerId
from exo.utils.channels import Receiver, Sender
from exo.utils.event_buffer import MultiSourceBuffer


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        local_event_receiver: Receiver[ForwarderEvent],
        global_event_sender: Sender[ForwarderEvent],
        download_command_sender: Sender[ForwarderDownloadCommand],
    ):
        self.state = State()
        self._tg: TaskGroup = anyio.create_task_group()
        self.node_id = node_id
        self.session_id = session_id
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.download_command_sender = download_command_sender
        self._multi_buffer = MultiSourceBuffer[NodeId, Event]()
        self._event_log = DiskEventLog(EXO_EVENT_LOG_DIR / "master")
        self._pending_traces: dict[TaskId, dict[int, list[TraceEventData]]] = {}
        self._expected_ranks: dict[TaskId, set[int]] = {}
        self._jaccl_pending: dict[InstanceId, dict[int, dict[RunnerId, bytes]]] = {}
        self._process_managers: Sequence[ProcessManager] = [
            InstanceHealthReconciler(),
            NodeTimeoutReconciler(),
            MetaInstanceReconciler(),
        ]

    async def run(self):
        logger.info("Starting Master")

        try:
            async with self._tg as tg:
                tg.start_soon(self._event_processor)
                tg.start_soon(self._command_processor)
                tg.start_soon(self._reconcile)
        finally:
            self._event_log.close()
            self.global_event_sender.close()
            self.local_event_receiver.close()
            self.command_receiver.close()

    async def shutdown(self):
        logger.info("Stopping Master")
        self._tg.cancel_scope.cancel()

    async def _command_processor(self) -> None:
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                try:
                    logger.info(f"Executing command: {forwarder_command.command}")

                    generated_events: list[Event] = []
                    command = forwarder_command.command
                    instance_task_counts: dict[InstanceId, int] = {}
                    match command:
                        case TestCommand():
                            pass
                        case TextGeneration():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=TextGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=available_instance_ids[0],
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                        case ImageGeneration():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
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

                            self.command_task_mapping[command.command_id] = task_id

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
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
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

                            self.command_task_mapping[command.command_id] = task_id

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
                        case DeleteInstance():
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            for cmd in cancel_unnecessary_downloads(
                                placement, self.state.downloads
                            ):
                                await self.download_command_sender.send(
                                    ForwarderDownloadCommand(
                                        origin=self.node_id, command=cmd
                                    )
                                )
                            generated_events.extend(transition_events)
                        case CreateMetaInstance():
                            logger.info(
                                f"Creating MetaInstance for {command.meta_instance.model_id}"
                                f" (min_nodes={command.meta_instance.min_nodes},"
                                f" sharding={command.meta_instance.sharding})"
                            )
                            # Apply immediately so self.state is fresh across
                            # the await below and the reconciler won't race.
                            await self._apply_and_broadcast(
                                MetaInstanceCreated(meta_instance=command.meta_instance)
                            )
                            # Immediate placement attempt for responsiveness
                            model_card = await ModelCard.load(
                                command.meta_instance.model_id
                            )
                            # Re-check: reconciler may have satisfied it during the await
                            meta_id = command.meta_instance.meta_instance_id
                            still_unsatisfied = any(
                                m.meta_instance_id == meta_id
                                for m in find_unsatisfied_meta_instances(
                                    self.state.meta_instances,
                                    self.state.instances,
                                    self.state.topology,
                                )
                            )
                            if still_unsatisfied:
                                result = try_place_for_meta_instance(
                                    command.meta_instance,
                                    model_card,
                                    self.state.topology,
                                    self.state.instances,
                                    self.state.node_memory,
                                    self.state.node_network,
                                )
                                generated_events.extend(result.events)
                                if result.error is not None:
                                    generated_events.append(
                                        MetaInstancePlacementFailed(
                                            meta_instance_id=meta_id,
                                            reason=result.error,
                                        )
                                    )
                        case DeleteMetaInstance():
                            backing_count = sum(
                                1
                                for inst in self.state.instances.values()
                                if inst.meta_instance_id == command.meta_instance_id
                            )
                            logger.info(
                                f"Deleting MetaInstance {command.meta_instance_id}"
                                f" (cascade-deleting {backing_count} backing instance(s))"
                            )
                            generated_events.append(
                                MetaInstanceDeleted(
                                    meta_instance_id=command.meta_instance_id
                                )
                            )
                            # Cascade-delete backing instances atomically,
                            # cancelling any active tasks first.
                            for iid, inst in self.state.instances.items():
                                if inst.meta_instance_id == command.meta_instance_id:
                                    for task in self.state.tasks.values():
                                        if (
                                            task.instance_id == iid
                                            and task.task_status
                                            in (
                                                TaskStatus.Pending,
                                                TaskStatus.Running,
                                            )
                                        ):
                                            generated_events.append(
                                                TaskStatusUpdated(
                                                    task_status=TaskStatus.Cancelled,
                                                    task_id=task.task_id,
                                                )
                                            )
                                    generated_events.append(
                                        InstanceDeleted(instance_id=iid)
                                    )
                        case PlaceInstance():
                            placement = place_instance(
                                command,
                                self.state.topology,
                                self.state.instances,
                                self.state.node_memory,
                                self.state.node_network,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            placement = add_instance_to_placements(
                                command,
                                self.state.topology,
                                self.state.instances,
                            )
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
                                task_id := self.command_task_mapping.get(
                                    command.cancelled_command_id
                                )
                            ) is not None:
                                generated_events.append(
                                    TaskStatusUpdated(
                                        task_status=TaskStatus.Cancelled,
                                        task_id=task_id,
                                    )
                                )
                        case TaskFinished():
                            generated_events.append(
                                TaskDeleted(
                                    task_id=self.command_task_mapping[
                                        command.finished_command_id
                                    ]
                                )
                            )
                            self.command_task_mapping.pop(
                                command.finished_command_id, None
                            )
                        case RequestEventLog():
                            # We should just be able to send everything, since other buffers will ignore old messages
                            # rate limit to 1000 at a time
                            end = min(command.since_idx + 1000, len(self._event_log))
                            for i, event in enumerate(
                                self._event_log.read_range(command.since_idx, end),
                                start=command.since_idx,
                            ):
                                await self._send_event(IndexedEvent(idx=i, event=event))
                    for event in generated_events:
                        await self._apply_and_broadcast(event)
                except ValueError as e:
                    logger.opt(exception=e).warning("Error in command processor")

    async def _apply_and_broadcast(self, event: Event) -> None:
        """Apply event to state, persist to disk, and broadcast to workers.

        State is updated synchronously (before any await), so callers can
        rely on ``self.state`` reflecting this event immediately after the
        call.  Python's cooperative scheduling guarantees no interleaving
        between the state read and write.
        """
        logger.debug(f"Master indexing event: {str(event)[:100]}")
        indexed = IndexedEvent(event=event, idx=len(self._event_log))
        self.state = apply(self.state, indexed)
        event._master_time_stamp = datetime.now(tz=timezone.utc)  # pyright: ignore[reportPrivateUsage]
        self._event_log.append(event)
        await self._send_event(indexed)

    async def _reconcile(self) -> None:
        while True:
            for pm in self._process_managers:
                events = await pm.reconcile(self.state)
                for event in events:
                    await self._apply_and_broadcast(event)
            await anyio.sleep(1)

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
                        await self._handle_traces_collected(event)
                        continue

                    if isinstance(event, JacclSideChannelData):
                        await self._apply_and_broadcast(event)
                        await self._handle_jaccl_side_channel(event)
                        continue

                    if isinstance(event, NodeGatheredInfo):
                        event.when = str(datetime.now(tz=timezone.utc))

                    await self._apply_and_broadcast(event)

    # This function is re-entrant, take care!
    async def _send_event(self, event: IndexedEvent):
        # Convenience method since this line is ugly
        await self.global_event_sender.send(
            ForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )

    async def _handle_traces_collected(self, event: TracesCollected) -> None:
        task_id = event.task_id
        if task_id not in self._pending_traces:
            self._pending_traces[task_id] = {}
        self._pending_traces[task_id][event.rank] = event.traces

        if (
            task_id in self._expected_ranks
            and set(self._pending_traces[task_id].keys())
            >= self._expected_ranks[task_id]
        ):
            await self._merge_and_save_traces(task_id)

    async def _merge_and_save_traces(self, task_id: TaskId) -> None:
        all_trace_data: list[TraceEventData] = []
        for trace_data in self._pending_traces[task_id].values():
            all_trace_data.extend(trace_data)

        await self._apply_and_broadcast(
            TracesMerged(task_id=task_id, traces=all_trace_data)
        )

        del self._pending_traces[task_id]
        if task_id in self._expected_ranks:
            del self._expected_ranks[task_id]

    async def _handle_jaccl_side_channel(self, event: JacclSideChannelData) -> None:
        """Accumulate SideChannel contributions; when all runners for an instance
        have submitted for the same sequence, emit JacclSideChannelGathered."""
        iid = event.instance_id
        seq = event.sequence

        if iid not in self._jaccl_pending:
            self._jaccl_pending[iid] = {}
        if seq not in self._jaccl_pending[iid]:
            self._jaccl_pending[iid][seq] = {}
        self._jaccl_pending[iid][seq][event.runner_id] = event.data

        instance = self.state.instances.get(iid)
        if instance is None:
            logger.warning(f"JacclSideChannelData for unknown instance {iid}")
            return

        expected_runners = set(instance.shard_assignments.runner_to_shard.keys())
        submitted = set(self._jaccl_pending[iid][seq].keys())

        logger.info(
            f"JACCL side channel: instance={iid} seq={seq} "
            f"submitted={len(submitted)}/{len(expected_runners)}"
        )

        if submitted >= expected_runners:
            gathered = dict(self._jaccl_pending[iid][seq])
            del self._jaccl_pending[iid][seq]
            if not self._jaccl_pending[iid]:
                del self._jaccl_pending[iid]

            await self._apply_and_broadcast(
                JacclSideChannelGathered(
                    instance_id=iid,
                    sequence=seq,
                    gathered_data=gathered,
                )
            )
