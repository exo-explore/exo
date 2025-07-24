import asyncio
import os
import threading
from logging import Logger
from pathlib import Path
from typing import List

from master.api import start_fastapi_server
from master.election_callback import ElectionCallbacks
from master.forwarder_supervisor import ForwarderSupervisor
from shared.apply import apply
from shared.db.sqlite.config import EventLogConfig
from shared.db.sqlite.connector import AsyncSQLiteEventStorage
from shared.db.sqlite.event_log_manager import EventLogManager
from shared.models.model_cards import MODEL_CARDS
from shared.models.model_meta import get_model_meta
from shared.types.common import NodeId
from shared.types.events import (
    ChunkGenerated,
    CommandId,
    InstanceCreated,
    TaskCreated,
)
from shared.types.events.chunks import TokenChunk
from shared.types.events.commands import (
    ChatCompletionCommand,
    Command,
    CreateInstanceCommand,
    DeleteInstanceCommand,
)
from shared.types.state import State
from shared.types.tasks import ChatCompletionTask, TaskId, TaskStatus, TaskType
from shared.types.worker.common import InstanceId
from shared.types.worker.instances import (
    InstanceParams,
    ShardAssignments,
    TypeOfInstance,
)
from shared.types.worker.runners import RunnerId
from shared.types.worker.shards import PartitionStrategy, PipelineShardMetadata


## TODO: Hook this up properly
async def fake_tokens_task(events_log: AsyncSQLiteEventStorage, command_id: CommandId):
    model_id = "testmodelabc"

    for i in range(10):
        await asyncio.sleep(0.1)

        # Create the event with proper types and consistent IDs
        chunk_event = ChunkGenerated(
            command_id=command_id,
            chunk=TokenChunk(
                command_id=command_id,  # Use the same task_id
                idx=i,
                model=model_id,   # Use the same model_id
                text=f'text{i}',
                token_id=i
            )
        )

        # ChunkGenerated needs to be cast to the expected BaseEvent type
        await events_log.append_events(
            [chunk_event],
            origin=NodeId()
        )

    await asyncio.sleep(0.1)

    # Create the event with proper types and consistent IDs
    chunk_event = ChunkGenerated(
        command_id=command_id,
        chunk=TokenChunk(
            command_id=command_id,  # Use the same task_id
            idx=11,
            model=model_id,   # Use the same model_id
            text=f'text{11}',
            token_id=11,
            finish_reason='stop'
        )
    )

    # ChunkGenerated needs to be cast to the expected BaseEvent type
    await events_log.append_events(
        [chunk_event],
        origin=NodeId()
    )

def get_node_id() -> NodeId:
    return NodeId() # TODO

class Master:
    def __init__(self, command_buffer: list[Command], global_events: AsyncSQLiteEventStorage, forwarder_binary_path: Path, logger: Logger):
        self.command_buffer = command_buffer
        self.global_events = global_events
        self.node_id = get_node_id()
        self.forwarder_supervisor = ForwarderSupervisor(
            forwarder_binary_path=forwarder_binary_path,
            logger=logger
        )
        self.election_callbacks = ElectionCallbacks(self.forwarder_supervisor, logger)
        self.logger = logger

    async def _get_state_snapshot(self) -> State:
        # TODO: for now start from scratch every time, but we can optimize this by keeping a snapshot on disk so we don't have to re-apply all events
        return State()

    async def run(self):
        self.state = await self._get_state_snapshot()

        # TODO: we should clean these up on shutdown
        await self.forwarder_supervisor.start_as_replica()
        if os.getenv('EXO_RUN_AS_REPLICA') in set(['TRUE', 'true', '1']):
            await self.election_callbacks.on_became_replica()
        else:
            await self.election_callbacks.on_became_master()

        while True:
            next_event = None
            # 1. process commands
            if len(self.command_buffer) > 0:
                # for now we do one command at a time
                next_command = self.command_buffer.pop(0)
                self.logger.info(f"got command: {next_command}")
                # TODO: validate the command
                match next_command:
                    case ChatCompletionCommand():
                        # 1. find a valid instance for this request, if none exists ERROR (TODO)
                        instance_id = InstanceId()
                        task_id = TaskId()
                        # 2. publish TaskCreated event (TODO)
                        next_event = TaskCreated(
                            task_id=task_id,
                            task=ChatCompletionTask(
                                task_id=task_id,
                                task_type=TaskType.CHAT_COMPLETION,
                                instance_id=instance_id,
                                task_status=TaskStatus.PENDING,
                                task_params=next_command.request_params
                            )
                        )
                    case DeleteInstanceCommand():
                        # TODO
                        pass
                    case CreateInstanceCommand():
                        if next_command.model_id not in MODEL_CARDS:
                            raise ValueError(f"Model {next_command.model_id} not supported.")

                        # TODO: we should also support models that aren't in MODEL_CARDS
                        # if it's in MODEL_CARDS, use ModelMetadata from there, otherwise interpret as a repo_id and get from huggingface
                        if next_command.model_id in MODEL_CARDS:
                            model_card = MODEL_CARDS[next_command.model_id]
                            model_meta = model_card.metadata
                        else:
                            model_meta = await get_model_meta(next_command.model_id)

                        # TODO: how do we actually schedule an instance? TODO: @@@@@@𝕾𝖊𝖙𝖍@@@@@@
                        next_event = InstanceCreated(
                            instance_id=InstanceId(),
                            instance_params=InstanceParams(
                                shard_assignments=ShardAssignments(
                                    model_id=next_command.model_id,
                                    runner_to_shard={
                                        RunnerId(): PipelineShardMetadata(
                                            model_meta=model_meta,
                                            partition_strategy=PartitionStrategy.pipeline,
                                            device_rank=0,
                                            world_size=1,
                                            start_layer=0,
                                            end_layer=0,
                                            n_layers=0
                                        )
                                    },
                                    node_to_runner={}
                                ),
                                hosts=[]
                            ),
                            instance_type=TypeOfInstance.ACTIVE,
                        )

                if next_event is not None:
                    await self.global_events.append_events([next_event], origin=self.node_id)

            # 2. get latest events
            events = await self.global_events.get_events_since(self.state.last_event_applied_idx)
            if len(events) == 0:
                await asyncio.sleep(0.01)
                continue

            # 3. for each event, apply it to the state
            for event_from_log in events:
                self.state = apply(self.state, event_from_log)



async def main():
    logger = Logger(name='master_logger')

    event_log_manager = EventLogManager(EventLogConfig(), logger=logger)
    await event_log_manager.initialize()
    global_events: AsyncSQLiteEventStorage = event_log_manager.global_events

    command_buffer: List[Command] = []

    api_thread = threading.Thread(
        target=start_fastapi_server,
        args=(
            command_buffer,
            global_events,
        ),
        daemon=True
    )
    api_thread.start()
    logger.info('Running FastAPI server in a separate thread. Listening on port 8000.')

    master = Master(command_buffer, global_events, forwarder_binary_path=Path("forwarder"), logger=logger)
    await master.run()

if __name__ == "__main__":
    asyncio.run(main())
