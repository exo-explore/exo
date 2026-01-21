from datetime import datetime, timezone
from typing import Sequence

import anyio
import pytest
from loguru import logger

from exo.master.main import Master
from exo.routing.router import get_node_id_keypair
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.api import ChatCompletionMessage, ChatCompletionTaskParams
from exo.shared.types.commands import (
    ChatCompletion,
    CommandId,
    ForwarderCommand,
    PlaceInstance,
)
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    ForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    NodeGatheredInfo,
    TaskCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryUsage,
)
from exo.shared.types.tasks import ChatCompletion as ChatCompletionTask
from exo.shared.types.tasks import TaskStatus
from exo.shared.types.worker.instances import (
    InstanceMeta,
    MlxRingInstance,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding
from exo.utils.channels import channel


@pytest.mark.asyncio
async def test_master():
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_peer_id().to_base58())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    ge_sender, global_event_receiver = channel[ForwarderEvent]()
    command_sender, co_receiver = channel[ForwarderCommand]()
    local_event_sender, le_receiver = channel[ForwarderEvent]()

    all_events: list[IndexedEvent] = []

    def _get_events() -> Sequence[IndexedEvent]:
        orig_events = global_event_receiver.collect()
        for e in orig_events:
            all_events.append(
                IndexedEvent(
                    event=e.event,
                    idx=len(all_events),  # origin=e.origin,
                )
            )
        return all_events

    master = Master(
        node_id,
        session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
    )
    logger.info("run the master")
    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        sender_node_id = NodeId(f"{keypair.to_peer_id().to_base58()}_sender")
        # inject a NodeGatheredInfo event
        logger.info("inject a NodeGatheredInfo event")
        await local_event_sender.send(
            ForwarderEvent(
                origin_idx=0,
                origin=sender_node_id,
                session=session_id,
                event=(
                    NodeGatheredInfo(
                        when=str(datetime.now(tz=timezone.utc)),
                        node_id=node_id,
                        info=MemoryUsage(
                            ram_total=Memory.from_bytes(678948 * 1024),
                            ram_available=Memory.from_bytes(678948 * 1024),
                            swap_total=Memory.from_bytes(0),
                            swap_available=Memory.from_bytes(0),
                        ),
                    )
                ),
            )
        )

        # wait for initial topology event
        logger.info("wait for initial topology event")
        while len(list(master.state.topology.list_nodes())) == 0:
            await anyio.sleep(0.001)
        while len(master.state.node_memory) == 0:
            await anyio.sleep(0.001)

        logger.info("inject a CreateInstance Command")
        await command_sender.send(
            ForwarderCommand(
                origin=node_id,
                command=(
                    PlaceInstance(
                        command_id=CommandId(),
                        model_card=ModelCard(
                            model_id=ModelId("llama-3.2-1b"),
                            n_layers=16,
                            storage_size=Memory.from_bytes(678948),
                            hidden_size=7168,
                            supports_tensor=True,
                            tasks=[ModelTask.TextGeneration],
                        ),
                        sharding=Sharding.Pipeline,
                        instance_meta=InstanceMeta.MlxRing,
                        min_nodes=1,
                    )
                ),
            )
        )
        logger.info("wait for an instance")
        while len(master.state.instances.keys()) == 0:
            await anyio.sleep(0.001)
        logger.info("inject a ChatCompletion Command")
        await command_sender.send(
            ForwarderCommand(
                origin=node_id,
                command=(
                    ChatCompletion(
                        command_id=CommandId(),
                        request_params=ChatCompletionTaskParams(
                            model="llama-3.2-1b",
                            messages=[
                                ChatCompletionMessage(
                                    role="user", content="Hello, how are you?"
                                )
                            ],
                        ),
                    )
                ),
            )
        )
        while len(_get_events()) < 3:
            await anyio.sleep(0.01)

        events = _get_events()
        assert len(events) == 3
        assert events[0].idx == 0
        assert events[1].idx == 1
        assert events[2].idx == 2
        assert isinstance(events[0].event, NodeGatheredInfo)
        assert isinstance(events[1].event, InstanceCreated)
        created_instance = events[1].event.instance
        assert isinstance(created_instance, MlxRingInstance)
        runner_id = list(created_instance.shard_assignments.runner_to_shard.keys())[0]
        # Validate the shard assignments
        expected_shard_assignments = ShardAssignments(
            model_id=ModelId("llama-3.2-1b"),
            runner_to_shard={
                (runner_id): PipelineShardMetadata(
                    start_layer=0,
                    end_layer=16,
                    n_layers=16,
                    model_card=ModelCard(
                        model_id=ModelId("llama-3.2-1b"),
                        n_layers=16,
                        storage_size=Memory.from_bytes(678948),
                        hidden_size=7168,
                        supports_tensor=True,
                        tasks=[ModelTask.TextGeneration],
                    ),
                    device_rank=0,
                    world_size=1,
                )
            },
            node_to_runner={node_id: runner_id},
        )
        assert created_instance.shard_assignments == expected_shard_assignments
        # For single-node, hosts_by_node should have one entry with self-binding
        assert len(created_instance.hosts_by_node) == 1
        assert node_id in created_instance.hosts_by_node
        assert len(created_instance.hosts_by_node[node_id]) == 1
        assert created_instance.hosts_by_node[node_id][0].ip == "0.0.0.0"
        assert created_instance.ephemeral_port > 0
        assert isinstance(events[2].event, TaskCreated)
        assert events[2].event.task.task_status == TaskStatus.Pending
        assert isinstance(events[2].event.task, ChatCompletionTask)
        assert events[2].event.task.task_params == ChatCompletionTaskParams(
            model="llama-3.2-1b",
            messages=[
                ChatCompletionMessage(role="user", content="Hello, how are you?")
            ],
        )

        await master.shutdown()
