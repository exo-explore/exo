from datetime import datetime, timezone
from typing import Sequence

import anyio
import pytest
from loguru import logger

from exo.master.deployments import place_deployment
from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    CommandId,
    CreateDeployment,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.common import DeploymentId, ModelId, SessionId, SystemId
from exo.shared.types.deployments import DeploymentSpec
from exo.shared.types.events import (
    DeploymentCreated,
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    LocalForwarderEvent,
    NodeGatheredInfo,
    RunnerStatusUpdated,
    StateSnapshot,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryUsage,
)
from exo.shared.types.worker.instances import (
    InstanceMeta,
    MlxRingInstance,
    ShardAssignments,
)
from exo.shared.types.worker.runners import RunnerReady
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding
from exo.utils.channels import channel
from exo.utils.info_gatherer.info_gatherer import NodeBackends


@pytest.mark.asyncio
async def test_master():
    node_id = get_node_zid()
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    ge_sender, global_event_receiver = channel[GlobalForwarderEvent]()
    command_sender, co_receiver = channel[ForwarderCommand]()
    local_event_sender, le_receiver = channel[LocalForwarderEvent]()
    fcds, _fcdr = channel[ForwarderDownloadCommand]()
    snapshot_sender, _snapshot_receiver = channel[StateSnapshot]()

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
        snapshot_sender=snapshot_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        download_command_sender=fcds,
        membership=FakeMembership(node_id),
    )
    logger.info("run the master")
    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # inject a NodeGatheredInfo event
        logger.info("inject a NodeGatheredInfo event")
        await local_event_sender.send(
            LocalForwarderEvent(
                origin_idx=0,
                origin=SystemId("Worker"),
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
        await local_event_sender.send(
            LocalForwarderEvent(
                origin_idx=1,
                origin=SystemId("Worker"),
                session=session_id,
                event=(
                    NodeGatheredInfo(
                        when=str(datetime.now(tz=timezone.utc)),
                        node_id=node_id,
                        info=NodeBackends(backends=[Backend.MlxMetal]),
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
        while len(master.state.node_backends) == 0:
            await anyio.sleep(0.001)

        logger.info("inject a CreateDeployment Command")
        deployment_id = DeploymentId()
        spec = DeploymentSpec(
            model_id=ModelId("llama-3.2-1b"),
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
        )
        card = ModelCard(
            model_id=ModelId("llama-3.2-1b"),
            n_layers=16,
            storage_size=Memory.from_bytes(678948),
            hidden_size=7168,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        )
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("API"),
                command=CreateDeployment(
                    command_id=CommandId(),
                    deployment_id=deployment_id,
                    spec=spec,
                    model_card=card,
                    instance=place_deployment(
                        deployment_id,
                        spec,
                        card,
                        master.state,
                        master.state.instances,
                        allowed_nodes={node_id},
                    ).instance,
                ),
            )
        )
        logger.info("wait for an instance")
        while len(master.state.instances.keys()) == 0:
            await anyio.sleep(0.001)
        # The worker loads it: the master only sends requests to a ready instance
        (instance,) = master.state.instances.values()
        (runner_id,) = instance.shard_assignments.runner_to_shard
        await local_event_sender.send(
            LocalForwarderEvent(
                origin_idx=2,
                origin=SystemId("Worker"),
                session=session_id,
                event=RunnerStatusUpdated(
                    runner_id=runner_id, runner_status=RunnerReady()
                ),
            )
        )
        while runner_id not in master.state.runners:
            await anyio.sleep(0.001)
        while len(_get_events()) < 5:
            await anyio.sleep(0.01)

        events = [
            event
            for event in _get_events()
            if not isinstance(event.event, RunnerStatusUpdated)
        ]
        # The master places instances; requests never go through it
        assert [event.idx for event in events] == [0, 1, 2, 3]
        assert isinstance(events[0].event, NodeGatheredInfo)
        assert isinstance(events[1].event, NodeGatheredInfo)
        assert isinstance(events[2].event, DeploymentCreated)
        assert events[2].event.deployment.deployment_id == deployment_id
        assert isinstance(events[3].event, InstanceCreated)
        created_instance = events[3].event.instance
        assert created_instance.deployment_id == deployment_id
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
                        backends=[Backend.MlxMetal],
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
        await master.shutdown()
