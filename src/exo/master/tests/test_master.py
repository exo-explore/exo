import asyncio
import tempfile
from logging import Logger
from pathlib import Path
from typing import List, Sequence

import pytest

from exo.master.main import Master
from exo.shared.db.sqlite.config import EventLogConfig
from exo.shared.db.sqlite.connector import AsyncSQLiteEventStorage
from exo.shared.db.sqlite.event_log_manager import EventLogManager
from exo.shared.keypair import Keypair
from exo.shared.logging import logger_test_install
from exo.shared.types.api import ChatCompletionMessage, ChatCompletionTaskParams
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, EventFromEventLog, Heartbeat, TaskCreated
from exo.shared.types.events._events import (
    InstanceCreated,
    NodePerformanceMeasured,
    TopologyNodeCreated,
)
from exo.shared.types.events.commands import (
    ChatCompletionCommand,
    Command,
    CommandId,
    CreateInstanceCommand,
)
from exo.shared.types.models import ModelMetadata
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.tasks import ChatCompletionTask, TaskStatus, TaskType
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceStatus,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PartitionStrategy, PipelineShardMetadata


def _create_forwarder_dummy_binary() -> Path:
    path = Path(tempfile.mktemp()) / "forwarder.bin"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"#!/bin/sh\necho dummy forwarder && sleep 1000000\n")
        path.chmod(0o755)
    return path


@pytest.mark.asyncio
async def test_master():
    logger = Logger(name="test_master_logger")
    logger_test_install(logger)
    event_log_manager = EventLogManager(EventLogConfig())
    await event_log_manager.initialize()
    global_events: AsyncSQLiteEventStorage = event_log_manager.global_events
    await global_events.delete_all_events()

    async def _get_events() -> Sequence[EventFromEventLog[Event]]:
        orig_events = await global_events.get_events_since(0)
        override_idx_in_log = 1
        events: List[EventFromEventLog[Event]] = []
        for e in orig_events:
            if isinstance(e.event, Heartbeat):
                continue
            events.append(
                EventFromEventLog(
                    event=e.event, origin=e.origin, idx_in_log=override_idx_in_log
                )
            )
            override_idx_in_log += 1
        return events

    command_buffer: List[Command] = []

    forwarder_binary_path = _create_forwarder_dummy_binary()

    node_id_keypair = Keypair.generate_ed25519()
    node_id = NodeId(node_id_keypair.to_peer_id().to_base58())
    master = Master(
        node_id_keypair,
        node_id,
        command_buffer=command_buffer,
        global_events=global_events,
        forwarder_binary_path=forwarder_binary_path,
        worker_events=global_events,
    )
    asyncio.create_task(master.run())
    # wait for initial topology event
    while len(list(master.state.topology.list_nodes())) == 0:
        print("waiting")
        await asyncio.sleep(0.001)
    # inject a NodePerformanceProfile event
    await event_log_manager.global_events.append_events(
        [
            NodePerformanceMeasured(
                node_id=node_id,
                node_profile=NodePerformanceProfile(
                    model_id="maccy",
                    chip_id="arm",
                    friendly_name="test",
                    memory=MemoryPerformanceProfile(
                        ram_total=678948 * 1024,
                        ram_available=678948 * 1024,
                        swap_total=0,
                        swap_available=0,
                    ),
                    network_interfaces=[],
                    system=SystemPerformanceProfile(flops_fp16=0),
                ),
            )
        ],
        origin=node_id,
    )
    while len(master.state.node_profiles) == 0:
        await asyncio.sleep(0.001)

    command_buffer.append(
        CreateInstanceCommand(
            command_id=CommandId(),
            instance_id=InstanceId(),
            model_meta=ModelMetadata(
                model_id="llama-3.2-1b",
                pretty_name="Llama 3.2 1B",
                n_layers=16,
                storage_size_kilobytes=678948,
            ),
        )
    )
    while len(master.state.instances.keys()) == 0:
        await asyncio.sleep(0.001)
    command_buffer.append(
        ChatCompletionCommand(
            command_id=CommandId(),
            request_params=ChatCompletionTaskParams(
                model="llama-3.2-1b",
                messages=[
                    ChatCompletionMessage(role="user", content="Hello, how are you?")
                ],
            ),
        )
    )
    while len(await _get_events()) < 4:
        await asyncio.sleep(0.001)

    events = await _get_events()
    print(events)
    assert len(events) == 4
    assert events[0].idx_in_log == 1
    assert isinstance(events[0].event, TopologyNodeCreated)
    assert isinstance(events[1].event, NodePerformanceMeasured)
    assert isinstance(events[2].event, InstanceCreated)
    runner_id = list(events[2].event.instance.shard_assignments.runner_to_shard.keys())[
        0
    ]
    assert events[2].event == InstanceCreated(
        instance=Instance(
            instance_id=events[2].event.instance.instance_id,
            instance_type=InstanceStatus.ACTIVE,
            shard_assignments=ShardAssignments(
                model_id="llama-3.2-1b",
                runner_to_shard={
                    (runner_id): PipelineShardMetadata(
                        partition_strategy=PartitionStrategy.pipeline,
                        start_layer=0,
                        end_layer=16,
                        n_layers=16,
                        model_meta=ModelMetadata(
                            model_id="llama-3.2-1b",
                            pretty_name="Llama 3.2 1B",
                            n_layers=16,
                            storage_size_kilobytes=678948,
                        ),
                        device_rank=0,
                        world_size=1,
                    )
                },
                node_to_runner={node_id: runner_id},
            ),
            hosts=[],
        )
    )
    assert isinstance(events[3].event, TaskCreated)
    assert events[3].event == TaskCreated(
        task_id=events[3].event.task_id,
        task=ChatCompletionTask(
            task_id=events[3].event.task_id,
            command_id=events[3].event.task.command_id,
            task_type=TaskType.CHAT_COMPLETION,
            instance_id=events[3].event.task.instance_id,
            task_status=TaskStatus.PENDING,
            task_params=ChatCompletionTaskParams(
                model="llama-3.2-1b",
                messages=[
                    ChatCompletionMessage(role="user", content="Hello, how are you?")
                ],
            ),
        ),
    )
    assert len(command_buffer) == 0
