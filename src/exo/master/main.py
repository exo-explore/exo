import asyncio
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import List

from exo.master.api import start_fastapi_server
from exo.master.election_callback import ElectionCallbacks
from exo.master.forwarder_supervisor import ForwarderRole, ForwarderSupervisor
from exo.master.placement import get_instance_placements, get_transition_events
from exo.shared.apply import apply
from exo.shared.db.sqlite.config import EventLogConfig
from exo.shared.db.sqlite.connector import AsyncSQLiteEventStorage
from exo.shared.db.sqlite.event_log_manager import EventLogManager
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    Event,
    Heartbeat,
    InstanceDeleted,
    TaskCreated,
    TaskDeleted,
    TopologyEdgeDeleted,
    TopologyNodeCreated,
)
from exo.shared.types.events.commands import (
    ChatCompletionCommand,
    Command,
    CreateInstanceCommand,
    DeleteInstanceCommand,
    TaskFinishedCommand,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import ChatCompletionTask, TaskId, TaskStatus, TaskType
from exo.shared.types.worker.instances import Instance
from exo.shared.utils import Keypair, get_node_id_keypair


class Master:
    def __init__(self, node_id_keypair: Keypair, node_id: NodeId, command_buffer: list[Command],
                 global_events: AsyncSQLiteEventStorage, worker_events: AsyncSQLiteEventStorage,
                 forwarder_binary_path: Path, logger: logging.Logger):
        self.state = State()
        self.node_id_keypair = node_id_keypair
        self.node_id = node_id
        self.command_buffer = command_buffer
        self.global_events = global_events
        self.worker_events = worker_events
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.forwarder_supervisor = ForwarderSupervisor(
            self.node_id,
            forwarder_binary_path=forwarder_binary_path,
            logger=logger
        )
        self.election_callbacks = ElectionCallbacks(self.forwarder_supervisor, logger)
        self.logger = logger

    @property
    def event_log_for_reads(self) -> AsyncSQLiteEventStorage:
        return self.global_events

    @property
    def event_log_for_writes(self) -> AsyncSQLiteEventStorage:
        if self.forwarder_supervisor.current_role == ForwarderRole.MASTER:
            return self.global_events
        else:
            return self.worker_events

    async def _get_state_snapshot(self) -> State:
        # TODO: for now start from scratch every time, but we can optimize this by keeping a snapshot on disk so we don't have to re-apply all events
        return State()

    async def _run_event_loop_body(self) -> None:
        next_events: list[Event] = []
        # 1. process commands
        if self.forwarder_supervisor.current_role == ForwarderRole.MASTER and len(self.command_buffer) > 0:
            # for now we do one command at a time
            next_command = self.command_buffer.pop(0)
            self.logger.info(f"got command: {next_command}")
            # TODO: validate the command
            match next_command:
                case ChatCompletionCommand():
                    matching_instance: Instance | None = None
                    for instance in self.state.instances.values():
                        if instance.shard_assignments.model_id == next_command.request_params.model:
                            matching_instance = instance
                            break
                    if not matching_instance:
                        raise ValueError(f"No instance found for model {next_command.request_params.model}")

                    task_id = TaskId()
                    next_events.append(TaskCreated(
                        task_id=task_id,
                        task=ChatCompletionTask(
                            task_type=TaskType.CHAT_COMPLETION,
                            task_id=task_id,
                            command_id=next_command.command_id,
                            instance_id=matching_instance.instance_id,
                            task_status=TaskStatus.PENDING,
                            task_params=next_command.request_params
                        )
                    ))

                    self.command_task_mapping[next_command.command_id] = task_id
                case DeleteInstanceCommand():
                    placement = get_instance_placements(next_command, self.state.topology, self.state.instances)
                    transition_events = get_transition_events(self.state.instances, placement)
                    next_events.extend(transition_events)
                case CreateInstanceCommand():
                    placement = get_instance_placements(next_command, self.state.topology, self.state.instances)
                    transition_events = get_transition_events(self.state.instances, placement)
                    next_events.extend(transition_events)
                case TaskFinishedCommand():
                    next_events.append(TaskDeleted(
                        task_id=self.command_task_mapping[next_command.command_id]
                    ))
                    del self.command_task_mapping[next_command.command_id]

            await self.event_log_for_writes.append_events(next_events, origin=self.node_id)
        # 2. get latest events
        events = await self.event_log_for_reads.get_events_since(self.state.last_event_applied_idx, ignore_no_op_events=True)
        if len(events) == 0:
            await asyncio.sleep(0.01)
            return
        self.logger.debug(f"got events: {events}")

        # 3. for each event, apply it to the state
        for event_from_log in events:
            self.logger.debug(f"applying event: {event_from_log}")
            self.state = apply(self.state, event_from_log)
        self.logger.debug(f"state: {self.state.model_dump_json()}")

        # TODO: This can be done in a better place. But for now, we use this to check if any running instances have been broken.
        write_events: list[Event] = []
        if any([isinstance(event_from_log.event, TopologyEdgeDeleted) for event_from_log in events]):
            connected_node_ids = set([x.node_id for x in self.state.topology.list_nodes()])
            for instance_id, instance in self.state.instances.items():
                delete = False
                for node_id in instance.shard_assignments.node_to_runner:
                    if node_id not in connected_node_ids:
                        delete = True
                        break
                if delete:
                    write_events.append(InstanceDeleted(
                        instance_id=instance_id
                    ))

        if write_events:
            await self.event_log_for_writes.append_events(events=write_events, origin=self.node_id)

    async def run(self):
        self.state = await self._get_state_snapshot()
        
        async def heartbeat_task():
            while True:
                await self.event_log_for_writes.append_events([Heartbeat(node_id=self.node_id)], origin=self.node_id)
                await asyncio.sleep(5)
        asyncio.create_task(heartbeat_task())

        # TODO: we should clean these up on shutdown
        await self.forwarder_supervisor.start_as_replica()
        if os.getenv('EXO_RUN_AS_REPLICA') in set(['TRUE', 'true', '1']):
            await self.election_callbacks.on_became_replica()
        else:
            await self.election_callbacks.on_became_master()

        role = "MASTER" if self.forwarder_supervisor.current_role == ForwarderRole.MASTER else "REPLICA"
        await self.event_log_for_writes.append_events([TopologyNodeCreated(node_id=self.node_id, role=role)], origin=self.node_id)
        while True:
            try:
                await self._run_event_loop_body()
            except Exception as e:
                self.logger.error(f"Error in _run_event_loop_body: {e}")
                traceback.print_exc()
                await asyncio.sleep(0.1)


async def async_main():
    logger = logging.getLogger('master_logger')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    node_id_keypair = get_node_id_keypair()
    node_id = NodeId(node_id_keypair.to_peer_id().to_base58())

    event_log_manager = EventLogManager(EventLogConfig(), logger=logger)
    await event_log_manager.initialize()
    global_events: AsyncSQLiteEventStorage = event_log_manager.global_events
    worker_events: AsyncSQLiteEventStorage = event_log_manager.worker_events

    command_buffer: List[Command] = []

    logger.info(f"Starting Master with node_id: {node_id}")

    api_thread = threading.Thread(
        target=start_fastapi_server,
        args=(
            command_buffer,
            global_events,
            lambda: master.state,
            "0.0.0.0",
            int(os.environ.get("API_PORT", 8000))
        ),
        daemon=True
    )
    api_thread.start()
    logger.info('Running FastAPI server in a separate thread. Listening on port 8000.')

    master = Master(node_id_keypair, node_id, command_buffer, global_events, worker_events,
                    Path(os.environ["GO_BUILD_DIR"])/"forwarder", logger)
    await master.run()

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
