from typing import Sequence

import anyio
import pytest
from loguru import logger

from exo.master.main import Master
from exo.routing.router import get_node_id_keypair
from exo.shared.types.api import ChatCompletionMessage, ChatCompletionTaskParams
from exo.shared.types.commands import (
    ChatCompletion,
    CommandId,
    CreateInstance,
    ForwarderCommand,
)
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    ForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    NodePerformanceMeasured,
    TaskCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.tasks import ChatCompletionTask, TaskStatus
from exo.shared.types.worker.instances import Instance, InstanceStatus, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
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
        tb_only=False,
    )
    logger.info("run the master")
    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        sender_node_id = NodeId(f"{keypair.to_peer_id().to_base58()}_sender")
        # inject a NodePerformanceProfile event
        logger.info("inject a NodePerformanceProfile event")
        await local_event_sender.send(
            ForwarderEvent(
                origin_idx=0,
                origin=sender_node_id,
                session=session_id,
                event=(
                    NodePerformanceMeasured(
                        node_id=node_id,
                        node_profile=NodePerformanceProfile(
                            model_id="maccy",
                            chip_id="arm",
                            friendly_name="test",
                            memory=MemoryPerformanceProfile(
                                ram_total=Memory.from_bytes(678948 * 1024),
                                ram_available=Memory.from_bytes(678948 * 1024),
                                swap_total=Memory.from_bytes(0),
                                swap_available=Memory.from_bytes(0),
                            ),
                            network_interfaces=[],
                            system=SystemPerformanceProfile(flops_fp16=0),
                        ),
                    )
                ),
            )
        )

        # wait for initial topology event
        logger.info("wait for initial topology event")
        while len(list(master.state.topology.list_nodes())) == 0:
            await anyio.sleep(0.001)
        while len(master.state.node_profiles) == 0:
            await anyio.sleep(0.001)

        logger.info("inject a CreateInstance Command")
        await command_sender.send(
            ForwarderCommand(
                origin=node_id,
                command=(
                    CreateInstance(
                        command_id=CommandId(),
                        model_meta=ModelMetadata(
                            model_id=ModelId("llama-3.2-1b"),
                            pretty_name="Llama 3.2 1B",
                            n_layers=16,
                            storage_size=Memory.from_bytes(678948),
                        ),
                        strategy="auto",
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
        assert isinstance(events[0].event, NodePerformanceMeasured)
        assert isinstance(events[1].event, InstanceCreated)
        runner_id = list(
            events[1].event.instance.shard_assignments.runner_to_shard.keys()
        )[0]
        assert events[1].event == InstanceCreated(
            event_id=events[1].event.event_id,
            instance=Instance(
                instance_id=events[1].event.instance.instance_id,
                instance_type=InstanceStatus.Active,
                shard_assignments=ShardAssignments(
                    model_id=ModelId("llama-3.2-1b"),
                    runner_to_shard={
                        (runner_id): PipelineShardMetadata(
                            start_layer=0,
                            end_layer=16,
                            n_layers=16,
                            model_meta=ModelMetadata(
                                model_id=ModelId("llama-3.2-1b"),
                                pretty_name="Llama 3.2 1B",
                                n_layers=16,
                                storage_size=Memory.from_bytes(678948),
                            ),
                            device_rank=0,
                            world_size=1,
                        )
                    },
                    node_to_runner={node_id: runner_id},
                ),
                hosts=[],
            ),
        )
        assert isinstance(events[2].event, TaskCreated)
        assert events[2].event == TaskCreated(
            event_id=events[2].event.event_id,
            task_id=events[2].event.task_id,
            task=ChatCompletionTask(
                task_id=events[2].event.task_id,
                command_id=events[2].event.task.command_id,
                instance_id=events[2].event.task.instance_id,
                task_status=TaskStatus.Pending,
                task_params=ChatCompletionTaskParams(
                    model="llama-3.2-1b",
                    messages=[
                        ChatCompletionMessage(
                            role="user", content="Hello, how are you?"
                        )
                    ],
                ),
            ),
        )
        await master.shutdown()
