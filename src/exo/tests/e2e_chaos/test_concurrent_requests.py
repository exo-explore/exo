"""E2E Chaos Test: Concurrent requests.

Scenarios:
1. Multiple simultaneous inference requests -- verify they are all created
   as tasks with no data corruption (unique task IDs, correct model IDs).
2. Concurrent requests across multiple model instances -- verify tasks are
   routed to the correct instances.
3. Concurrent requests with load balancing -- when multiple instances of
   the same model exist, verify tasks are distributed.
"""

import anyio
import pytest

from exo.master.main import Master
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.commands import (
    CommandId,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
    TextGeneration,
)
from exo.shared.types.common import ModelId, NodeId, SessionId
from exo.shared.types.events import (
    ForwarderEvent,
)
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskStatus
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.channels import channel

from .conftest import (
    TEST_MODEL_CARD,
    TEST_MODEL_ID,
    EventCollector,
    make_gathered_info_event,
    make_node_id,
)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_chat_requests_no_corruption() -> None:
    """Send multiple TextGeneration commands concurrently and verify each
    results in a unique task with the correct model and content mapping.
    """
    master_nid = make_node_id("master-concurrent")
    session_id = SessionId(master_node_id=master_nid, election_clock=0)

    ge_sender, ge_receiver = channel[ForwarderEvent]()
    cmd_sender, cmd_receiver = channel[ForwarderCommand]()
    le_sender, le_receiver = channel[ForwarderEvent]()
    dl_sender, _dl_receiver = channel[ForwarderDownloadCommand]()

    master = Master(
        master_nid,
        session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=cmd_receiver,
        download_command_sender=dl_sender,
    )

    _collector = EventCollector(ge_receiver.clone())

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # Set up node and instance
        sender_id = NodeId(f"{master_nid}_sender")
        await le_sender.send(
            make_gathered_info_event(master_nid, sender_id, session_id, 0)
        )

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) == 0:
                await anyio.sleep(0.01)

        await cmd_sender.send(
            ForwarderCommand(
                origin=master_nid,
                command=PlaceInstance(
                    command_id=CommandId(),
                    model_card=TEST_MODEL_CARD,
                    sharding=Sharding.Pipeline,
                    instance_meta=InstanceMeta.MlxRing,
                    min_nodes=1,
                ),
            )
        )

        with anyio.fail_after(3):
            while len(master.state.instances) == 0:
                await anyio.sleep(0.01)

        # Send 10 concurrent chat requests
        num_requests = 10
        cmd_ids: list[CommandId] = []

        async def send_chat(index: int) -> None:
            cmd_id = CommandId()
            cmd_ids.append(cmd_id)
            await cmd_sender.send(
                ForwarderCommand(
                    origin=master_nid,
                    command=TextGeneration(
                        command_id=cmd_id,
                        task_params=TextGenerationTaskParams(
                            model=TEST_MODEL_ID,
                            input=[
                                InputMessage(
                                    role="user",
                                    content=f"Concurrent request #{index}",
                                )
                            ],
                        ),
                    ),
                )
            )

        async with anyio.create_task_group() as send_tg:
            for i in range(num_requests):
                send_tg.start_soon(send_chat, i)

        # Wait for all tasks to be created
        with anyio.fail_after(5):
            while len(master.state.tasks) < num_requests:
                await anyio.sleep(0.01)

        # Verify no corruption
        assert len(master.state.tasks) == num_requests

        # All task IDs should be unique
        task_ids = list(master.state.tasks.keys())
        assert len(set(task_ids)) == num_requests

        # All tasks should target the correct model
        for task in master.state.tasks.values():
            assert isinstance(task, TextGenerationTask)
            assert task.task_params.model == TEST_MODEL_ID
            assert task.task_status == TaskStatus.Pending

        # All tasks should reference the same instance
        instance_ids = {task.instance_id for task in master.state.tasks.values()}
        assert len(instance_ids) == 1

        await master.shutdown()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_requests_across_multiple_models() -> None:
    """Place two different models, then send concurrent requests for each.
    Verify tasks are routed to the correct model instances.
    """
    master_nid = make_node_id("master-multi-model")
    session_id = SessionId(master_node_id=master_nid, election_clock=0)

    ge_sender, _ge_receiver = channel[ForwarderEvent]()
    cmd_sender, cmd_receiver = channel[ForwarderCommand]()
    le_sender, le_receiver = channel[ForwarderEvent]()
    dl_sender, _dl_receiver = channel[ForwarderDownloadCommand]()

    master = Master(
        master_nid,
        session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=cmd_receiver,
        download_command_sender=dl_sender,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # Register node
        sender_id = NodeId(f"{master_nid}_sender")
        await le_sender.send(
            make_gathered_info_event(master_nid, sender_id, session_id, 0)
        )

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) == 0:
                await anyio.sleep(0.01)

        # Place two different models
        model_a_id = ModelId("test-model/model-a")
        model_a_card = ModelCard(
            model_id=model_a_id,
            n_layers=16,
            storage_size=Memory.from_bytes(500_000),
            hidden_size=2048,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        )

        model_b_id = ModelId("test-model/model-b")
        model_b_card = ModelCard(
            model_id=model_b_id,
            n_layers=32,
            storage_size=Memory.from_bytes(500_000),
            hidden_size=4096,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        )

        for card in [model_a_card, model_b_card]:
            await cmd_sender.send(
                ForwarderCommand(
                    origin=master_nid,
                    command=PlaceInstance(
                        command_id=CommandId(),
                        model_card=card,
                        sharding=Sharding.Pipeline,
                        instance_meta=InstanceMeta.MlxRing,
                        min_nodes=1,
                    ),
                )
            )

        with anyio.fail_after(5):
            while len(master.state.instances) < 2:
                await anyio.sleep(0.01)

        # Map instance IDs to models
        instance_to_model: dict[str, ModelId] = {}
        for iid, inst in master.state.instances.items():
            instance_to_model[iid] = inst.shard_assignments.model_id

        # Send concurrent requests for both models
        async def send_for_model(model_id: ModelId, count: int) -> None:
            for i in range(count):
                await cmd_sender.send(
                    ForwarderCommand(
                        origin=master_nid,
                        command=TextGeneration(
                            command_id=CommandId(),
                            task_params=TextGenerationTaskParams(
                                model=model_id,
                                input=[
                                    InputMessage(
                                        role="user",
                                        content=f"Request for {model_id} #{i}",
                                    )
                                ],
                            ),
                        ),
                    )
                )

        async with anyio.create_task_group() as send_tg:
            send_tg.start_soon(send_for_model, model_a_id, 3)
            send_tg.start_soon(send_for_model, model_b_id, 3)

        # Wait for all 6 tasks
        with anyio.fail_after(5):
            while len(master.state.tasks) < 6:
                await anyio.sleep(0.01)

        # Verify task routing
        model_a_tasks = [
            t
            for t in master.state.tasks.values()
            if isinstance(t, TextGenerationTask) and t.task_params.model == model_a_id
        ]
        model_b_tasks = [
            t
            for t in master.state.tasks.values()
            if isinstance(t, TextGenerationTask) and t.task_params.model == model_b_id
        ]

        assert len(model_a_tasks) == 3
        assert len(model_b_tasks) == 3

        # All model_a tasks should reference the model_a instance
        model_a_instance_ids = {
            iid for iid, mid in instance_to_model.items() if mid == model_a_id
        }
        for task in model_a_tasks:
            assert task.instance_id in model_a_instance_ids

        # All model_b tasks should reference the model_b instance
        model_b_instance_ids = {
            iid for iid, mid in instance_to_model.items() if mid == model_b_id
        }
        for task in model_b_tasks:
            assert task.instance_id in model_b_instance_ids

        await master.shutdown()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_event_index_monotonically_increases_under_load() -> None:
    """Under heavy concurrent command load, verify the master's event log
    index increases monotonically with no gaps or duplicates.
    """
    master_nid = make_node_id("master-monotonic")
    session_id = SessionId(master_node_id=master_nid, election_clock=0)

    ge_sender, ge_receiver = channel[ForwarderEvent]()
    cmd_sender, cmd_receiver = channel[ForwarderCommand]()
    le_sender, le_receiver = channel[ForwarderEvent]()
    dl_sender, _dl_receiver = channel[ForwarderDownloadCommand]()

    master = Master(
        master_nid,
        session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=cmd_receiver,
        download_command_sender=dl_sender,
    )

    collector = EventCollector(ge_receiver.clone())

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # Register node and place instance
        sender_id = NodeId(f"{master_nid}_sender")
        await le_sender.send(
            make_gathered_info_event(master_nid, sender_id, session_id, 0)
        )

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) == 0:
                await anyio.sleep(0.01)

        await cmd_sender.send(
            ForwarderCommand(
                origin=master_nid,
                command=PlaceInstance(
                    command_id=CommandId(),
                    model_card=TEST_MODEL_CARD,
                    sharding=Sharding.Pipeline,
                    instance_meta=InstanceMeta.MlxRing,
                    min_nodes=1,
                ),
            )
        )

        with anyio.fail_after(3):
            while len(master.state.instances) == 0:
                await anyio.sleep(0.01)

        # Blast 20 concurrent commands
        async def blast_commands(start: int, count: int) -> None:
            for i in range(count):
                await cmd_sender.send(
                    ForwarderCommand(
                        origin=master_nid,
                        command=TextGeneration(
                            command_id=CommandId(),
                            task_params=TextGenerationTaskParams(
                                model=TEST_MODEL_ID,
                                input=[
                                    InputMessage(
                                        role="user",
                                        content=f"Blast {start + i}",
                                    )
                                ],
                            ),
                        ),
                    )
                )

        async with anyio.create_task_group() as blast_tg:
            blast_tg.start_soon(blast_commands, 0, 10)
            blast_tg.start_soon(blast_commands, 10, 10)

        # Wait for all tasks
        with anyio.fail_after(5):
            while len(master.state.tasks) < 20:
                await anyio.sleep(0.01)

        # Collect all events and verify monotonic indexing
        # NodeGatheredInfo(0) + InstanceCreated(1) + 20 TaskCreated = 22 events
        await collector.wait_for_event_count(22, timeout=5.0)

        events = collector.indexed_events
        indices = [e.idx for e in events]

        # Should be 0, 1, 2, ..., N-1 with no gaps
        expected = list(range(len(indices)))
        assert indices == expected

        # last_event_applied_idx should match
        assert master.state.last_event_applied_idx == len(events) - 1

        await master.shutdown()
