"""Concurrent routing tests for the master.

These tests pin the master's TextGeneration routing behavior under burst load:
when M concurrent requests arrive for a model that has N running instances, the
greedy lowest-in-flight-count selector should distribute the M requests
ceil(M/N)-balanced across the N instances. Without correct serialization between
the command processor and the event processor, all M requests would see "0
in-flight tasks" for the same instance and pile onto it.

This is the regression that commit cb1b563b ("index master events synchronously
to enable multi-instance load balancing") fixed. These tests lock that behavior
in.
"""

from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Sequence

import anyio
import pytest

from exo.master.main import Master
from exo.routing.router import get_node_id_keypair
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.commands import (
    CommandId,
    ForwarderCommand,
    ForwarderDownloadCommand,
    TextGeneration,
)
from exo.shared.types.common import (
    Host,
    ModelId,
    NodeId,
    SessionId,
    SystemId,
)
from exo.shared.types.events import (
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    LocalForwarderEvent,
    NodeGatheredInfo,
    TaskCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import channel


def _make_model_card(model_id: str) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        n_layers=16,
        storage_size=Memory.from_bytes(1024),
        hidden_size=128,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


def _make_instance_created(
    model_id: str, node_id: NodeId, port: int
) -> InstanceCreated:
    """Build a synthetic InstanceCreated event for a single-node Pipeline-1
    instance of `model_id`. Each call returns a fresh InstanceId/RunnerId so
    multiple instances of the same model can coexist."""
    instance_id = InstanceId()
    runner_id = RunnerId()
    model_card = _make_model_card(model_id)
    shard = PipelineShardMetadata(
        start_layer=0,
        end_layer=16,
        n_layers=16,
        model_card=model_card,
        device_rank=0,
        world_size=1,
    )
    instance = MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=ModelId(model_id),
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={node_id: [Host(ip="127.0.0.1", port=port)]},
        ephemeral_port=port,
    )
    return InstanceCreated(instance=instance)


async def _wait_for(
    predicate: Callable[[], bool],
    timeout_s: float = 5.0,
    interval_s: float = 0.005,
) -> None:
    deadline = anyio.current_time() + timeout_s
    while not predicate():
        if anyio.current_time() > deadline:
            raise TimeoutError("predicate did not become true within timeout")
        await anyio.sleep(interval_s)


@pytest.mark.asyncio
async def test_concurrent_text_generation_distributes_across_instances():
    """When M concurrent TextGeneration commands arrive for a model with N
    running instances, the master should route them so each instance receives
    exactly ceil(M/N) or floor(M/N) tasks — never all-to-one."""
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    ge_sender, global_event_receiver = channel[GlobalForwarderEvent]()
    command_sender, co_receiver = channel[ForwarderCommand]()
    local_event_sender, le_receiver = channel[LocalForwarderEvent]()
    fcds, _fcdr = channel[ForwarderDownloadCommand]()

    indexed_events: list[IndexedEvent] = []

    def _drain_events() -> Sequence[IndexedEvent]:
        new = global_event_receiver.collect()
        for e in new:
            indexed_events.append(IndexedEvent(event=e.event, idx=len(indexed_events)))
        return indexed_events

    master = Master(
        node_id,
        session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        download_command_sender=fcds,
    )

    model_id = "test/concurrent-routing-model"
    instance_count = 4
    request_count = 12  # 3 tasks per instance when balanced
    instance_ids: list[InstanceId] = []
    events: Sequence[IndexedEvent] = []

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # 1. Register the node so the master accepts events from it.
        await local_event_sender.send(
            LocalForwarderEvent(
                origin_idx=0,
                origin=SystemId("Worker"),
                session=session_id,
                event=NodeGatheredInfo(
                    when=str(datetime.now(tz=timezone.utc)),
                    node_id=node_id,
                    info=MemoryUsage(
                        ram_total=Memory.from_bytes(1024 * 1024 * 1024),
                        ram_available=Memory.from_bytes(1024 * 1024 * 1024),
                        swap_total=Memory.from_bytes(0),
                        swap_available=Memory.from_bytes(0),
                    ),
                ),
            )
        )

        # 2. Inject N synthetic InstanceCreated events. Each gets a unique
        #    InstanceId/RunnerId so they coexist in state.instances.
        for i in range(instance_count):
            event = _make_instance_created(model_id, node_id, port=10_000 + i)
            instance_ids.append(event.instance.instance_id)
            await local_event_sender.send(
                LocalForwarderEvent(
                    origin_idx=i + 1,
                    origin=SystemId("Worker"),
                    session=session_id,
                    event=event,
                )
            )

        # Wait until the master has applied all N InstanceCreated events.
        await _wait_for(lambda: len(master.state.instances) == instance_count)

        # 3. Fire M concurrent TextGeneration commands. Sending via send()
        #    in a tight loop is enough to queue them faster than the command
        #    processor can drain — which is the contention pattern that
        #    pre-cb1b563b regressed on.
        for i in range(request_count):
            await command_sender.send(
                ForwarderCommand(
                    origin=SystemId("API"),
                    command=TextGeneration(
                        command_id=CommandId(),
                        task_params=TextGenerationTaskParams(
                            model=ModelId(model_id),
                            input=[InputMessage(role="user", content=f"req {i}")],
                        ),
                    ),
                )
            )

        # 4. Wait for all M TaskCreated events to land. Total expected
        #    indexed events = 1 NodeGatheredInfo + N InstanceCreated + M TaskCreated.
        expected_total = 1 + instance_count + request_count
        await _wait_for(lambda: len(_drain_events()) >= expected_total, timeout_s=10.0)

        events = _drain_events()
        await master.shutdown()

    task_created_events: list[TaskCreated] = [
        e.event for e in events if isinstance(e.event, TaskCreated)
    ]
    assert len(task_created_events) == request_count, (
        f"expected {request_count} TaskCreated events, got {len(task_created_events)}"
    )

    # Every TaskCreated must reference one of our injected instances.
    instance_id_set = set(instance_ids)
    distribution: Counter[InstanceId] = Counter()
    for tc in task_created_events:
        task = tc.task
        assert isinstance(task, TextGenerationTask)
        assert task.instance_id in instance_id_set, (
            f"task routed to unknown instance {task.instance_id}"
        )
        distribution[task.instance_id] += 1

    # Perfect-balance assertions: with greedy lowest-count routing and no
    # tasks ever marked complete, the count per instance must be either
    # floor(M/N) or ceil(M/N).
    expected_min = request_count // instance_count
    expected_max = -(-request_count // instance_count)  # ceil
    counts = list(distribution.values())
    assert len(distribution) == instance_count, (
        f"expected all {instance_count} instances to receive at least one task, "
        f"got {len(distribution)} ({distribution})"
    )
    assert min(counts) >= expected_min, (
        f"distribution {dict(distribution)} has an instance below the floor "
        f"({expected_min}) — load is unbalanced"
    )
    assert max(counts) <= expected_max, (
        f"distribution {dict(distribution)} has an instance above the ceiling "
        f"({expected_max}) — pile-on regression"
    )


@pytest.mark.asyncio
async def test_text_generation_does_not_cross_model_boundaries():
    """Two models with overlapping instances must not route across each other.
    Requests for model A only go to instances of model A, and vice versa."""
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    ge_sender, global_event_receiver = channel[GlobalForwarderEvent]()
    command_sender, co_receiver = channel[ForwarderCommand]()
    local_event_sender, le_receiver = channel[LocalForwarderEvent]()
    fcds, _fcdr = channel[ForwarderDownloadCommand]()

    indexed_events: list[IndexedEvent] = []

    def _drain_events() -> Sequence[IndexedEvent]:
        new = global_event_receiver.collect()
        for e in new:
            indexed_events.append(IndexedEvent(event=e.event, idx=len(indexed_events)))
        return indexed_events

    master = Master(
        node_id,
        session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        download_command_sender=fcds,
    )

    model_a = "test/model-a"
    model_b = "test/model-b"
    a_ids: set[InstanceId] = set()
    b_ids: set[InstanceId] = set()
    events: Sequence[IndexedEvent] = []

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        await local_event_sender.send(
            LocalForwarderEvent(
                origin_idx=0,
                origin=SystemId("Worker"),
                session=session_id,
                event=NodeGatheredInfo(
                    when=str(datetime.now(tz=timezone.utc)),
                    node_id=node_id,
                    info=MemoryUsage(
                        ram_total=Memory.from_bytes(1024 * 1024 * 1024),
                        ram_available=Memory.from_bytes(1024 * 1024 * 1024),
                        swap_total=Memory.from_bytes(0),
                        swap_available=Memory.from_bytes(0),
                    ),
                ),
            )
        )

        # 2 instances of A, 3 instances of B.
        port = 10_000
        origin_idx = 1
        for _ in range(2):
            ev = _make_instance_created(model_a, node_id, port=port)
            a_ids.add(ev.instance.instance_id)
            await local_event_sender.send(
                LocalForwarderEvent(
                    origin_idx=origin_idx,
                    origin=SystemId("Worker"),
                    session=session_id,
                    event=ev,
                )
            )
            origin_idx += 1
            port += 1
        for _ in range(3):
            ev = _make_instance_created(model_b, node_id, port=port)
            b_ids.add(ev.instance.instance_id)
            await local_event_sender.send(
                LocalForwarderEvent(
                    origin_idx=origin_idx,
                    origin=SystemId("Worker"),
                    session=session_id,
                    event=ev,
                )
            )
            origin_idx += 1
            port += 1

        await _wait_for(lambda: len(master.state.instances) == 5)

        # 4 requests for A, 6 for B.
        for i in range(4):
            await command_sender.send(
                ForwarderCommand(
                    origin=SystemId("API"),
                    command=TextGeneration(
                        command_id=CommandId(),
                        task_params=TextGenerationTaskParams(
                            model=ModelId(model_a),
                            input=[InputMessage(role="user", content=f"a{i}")],
                        ),
                    ),
                )
            )
        for i in range(6):
            await command_sender.send(
                ForwarderCommand(
                    origin=SystemId("API"),
                    command=TextGeneration(
                        command_id=CommandId(),
                        task_params=TextGenerationTaskParams(
                            model=ModelId(model_b),
                            input=[InputMessage(role="user", content=f"b{i}")],
                        ),
                    ),
                )
            )

        expected_total = (
            1 + 5 + 4 + 6
        )  # NodeGatheredInfo + 5 InstanceCreated + 10 TaskCreated
        await _wait_for(lambda: len(_drain_events()) >= expected_total, timeout_s=10.0)

        events = _drain_events()
        await master.shutdown()

    task_created_events: list[TaskCreated] = [
        e.event for e in events if isinstance(e.event, TaskCreated)
    ]
    assert len(task_created_events) == 10

    a_distribution: Counter[InstanceId] = Counter()
    b_distribution: Counter[InstanceId] = Counter()
    for tc in task_created_events:
        task = tc.task
        assert isinstance(task, TextGenerationTask)
        if task.task_params.model == ModelId(model_a):
            assert task.instance_id in a_ids, "model A request routed to non-A instance"
            a_distribution[task.instance_id] += 1
        elif task.task_params.model == ModelId(model_b):
            assert task.instance_id in b_ids, "model B request routed to non-B instance"
            b_distribution[task.instance_id] += 1
        else:
            pytest.fail(f"unexpected model {task.task_params.model}")

    # 4 / 2 = 2 each for A.
    assert sum(a_distribution.values()) == 4
    assert all(c == 2 for c in a_distribution.values()), (
        f"model A unbalanced: {dict(a_distribution)}"
    )
    # 6 / 3 = 2 each for B.
    assert sum(b_distribution.values()) == 6
    assert all(c == 2 for c in b_distribution.values()), (
        f"model B unbalanced: {dict(b_distribution)}"
    )
