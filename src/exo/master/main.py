from datetime import datetime, timezone

from anyio import create_task_group
from anyio.abc import TaskGroup
from loguru import logger

from exo.master.placement import (
    get_instance_placements_after_create,
    get_instance_placements_after_delete,
    get_transition_events,
)
from exo.shared.apply import apply
from exo.shared.types.commands import (
    ChatCompletion,
    CreateInstance,
    DeleteInstance,
    ForwarderCommand,
    KillCommand,
    RequestEventLog,
    TaskFinished,
    TestCommand,
)
from exo.shared.types.common import CommandId, NodeId, SessionId
from exo.shared.types.events import (
    Event,
    ForwarderEvent,
    IndexedEvent,
    InstanceDeleted,
    TaskCreated,
    TaskDeleted,
    TopologyEdgeDeleted,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ChatCompletion as ChatCompletionTask,
)
from exo.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.event_buffer import MultiSourceBuffer


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        # Receiving indexed events from the forwarder to be applied to state
        # Ideally these would be WorkerForwarderEvents but type system says no :(
        local_event_receiver: Receiver[ForwarderEvent],
        # Send events to the forwarder to be indexed (usually from command processing)
        # Ideally these would be MasterForwarderEvents but type system says no :(
        global_event_sender: Sender[ForwarderEvent],
        tb_only: bool = False,
    ):
        self.state = State()
        self._tg: TaskGroup | None = None
        self.node_id = node_id
        self.session_id = session_id
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        send, recv = channel[Event]()
        self.event_sender: Sender[Event] = send
        self._loopback_event_receiver: Receiver[Event] = recv
        self._loopback_event_sender: Sender[ForwarderEvent] = (
            local_event_receiver.clone_sender()
        )
        self._multi_buffer = MultiSourceBuffer[NodeId, Event]()
        # TODO: not have this
        self._event_log: list[Event] = []
        self.tb_only = tb_only

    async def run(self):
        logger.info("Starting Master")

        async with create_task_group() as tg:
            self._tg = tg
            tg.start_soon(self._event_processor)
            tg.start_soon(self._command_processor)
            tg.start_soon(self._loopback_processor)
        self.global_event_sender.close()
        self.local_event_receiver.close()
        self.command_receiver.close()
        self._loopback_event_sender.close()
        self._loopback_event_receiver.close()

    async def shutdown(self):
        if self._tg:
            logger.info("Stopping Master")
            self._tg.cancel_scope.cancel()

    async def _command_processor(self) -> None:
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                try:
                    logger.info(f"Executing command: {forwarder_command.command}")
                    generated_events: list[Event] = []
                    command = forwarder_command.command
                    match command:
                        case TestCommand() | KillCommand():
                            pass
                        case ChatCompletion():
                            instance_task_counts: dict[InstanceId, int] = {}
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.request_params.model
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
                                    f"No instance found for model {command.request_params.model}"
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
                                    task=ChatCompletionTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=available_instance_ids[0],
                                        task_status=TaskStatus.Pending,
                                        task_params=command.request_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                        case DeleteInstance():
                            placement = get_instance_placements_after_delete(
                                command, self.state.instances
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            placement = get_instance_placements_after_create(
                                command,
                                self.state.topology,
                                self.state.instances,
                                tb_only=self.tb_only,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement
                            )
                            generated_events.extend(transition_events)
                        case TaskFinished():
                            generated_events.append(
                                TaskDeleted(
                                    task_id=self.command_task_mapping[
                                        command.finished_command_id
                                    ]
                                )
                            )
                            if command.finished_command_id in self.command_task_mapping:
                                del self.command_task_mapping[
                                    command.finished_command_id
                                ]
                        case RequestEventLog():
                            # We should just be able to send everything, since other buffers will ignore old messages
                            for i in range(command.since_idx, len(self._event_log)):
                                await self._send_event(
                                    IndexedEvent(idx=i, event=self._event_log[i])
                                )
                    for event in generated_events:
                        await self.event_sender.send(event)
                except ValueError as e:
                    logger.opt(exception=e).warning("Error in command processor")

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
                    indexed = IndexedEvent(event=event, idx=len(self._event_log))
                    self.state = apply(self.state, indexed)

                    event._master_time_stamp = datetime.now(tz=timezone.utc)  # pyright: ignore[reportPrivateUsage]

                    # TODO: SQL <- What does this mean?
                    self._event_log.append(event)
                    await self._send_event(indexed)

                    # TODO: This can be done in a better place. But for now, we use this to check if any running instances have been broken.
                    if isinstance(event, TopologyEdgeDeleted):
                        connected_node_ids = set(
                            [x.node_id for x in self.state.topology.list_nodes()]
                        )
                        for instance_id, instance in self.state.instances.items():
                            for node_id in instance.shard_assignments.node_to_runner:
                                if node_id not in connected_node_ids:
                                    await self.event_sender.send(
                                        InstanceDeleted(instance_id=instance_id)
                                    )
                                    break

    async def _loopback_processor(self) -> None:
        # this would ideally not be necessary.
        # this is WAY less hacky than how I was working around this before
        local_index = 0
        with self._loopback_event_receiver as events:
            async for event in events:
                await self._loopback_event_sender.send(
                    ForwarderEvent(
                        origin=NodeId(f"master_{self.node_id}"),
                        origin_idx=local_index,
                        session=self.session_id,
                        event=event,
                    )
                )
                local_index += 1

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
