"""A real master keeping a deployment's model running, with its events going round through a
stand-in for the event router as they do in a node."""

from collections.abc import Callable
from datetime import datetime, timezone

import anyio
import pytest

from exo.master import keeper
from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    Command,
    CommandId,
    CreateDeployment,
    DeleteDeployment,
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.common import ModelId, SessionId, SystemId
from exo.shared.types.deployments import Deployment, DeploymentId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    NodeGatheredInfo,
    RunnerStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.runners import RunnerReady
from exo.shared.types.worker.shards import Sharding
from exo.utils.channels import channel
from exo.utils.info_gatherer.info_gatherer import NodeBackends

MODEL = ModelCard(
    model_id=ModelId("test/model-a"),
    n_layers=16,
    storage_size=Memory.from_bytes(678948),
    hidden_size=7168,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)


async def _until(condition: Callable[[], bool]) -> None:
    with anyio.fail_after(10):
        while not condition():
            await anyio.sleep(0.01)


@pytest.mark.asyncio
async def test_master_keeps_a_deployed_model_running(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(keeper, "STARTUP_GRACE", 0.0)
    node_id = get_node_zid()
    session_id = SessionId(master_node_id=node_id, election_clock=0)
    global_sender, _global_receiver = channel[GlobalForwarderEvent]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    local_sender, local_receiver = channel[LocalForwarderEvent]()
    download_sender, _download_receiver = channel[ForwarderDownloadCommand]()
    event_sender, event_receiver = channel[Event]()

    async def event_router() -> None:
        # The master's own events come back to it as local events, as through the real router
        origin = SystemId()
        with event_receiver as events:
            index = 0
            async for event in events:
                await local_sender.send(
                    LocalForwarderEvent(
                        origin=origin, origin_idx=index, session=session_id, event=event
                    )
                )
                index += 1

    worker_index = 0

    async def worker_reports(event: Event) -> None:
        nonlocal worker_index
        await local_sender.send(
            LocalForwarderEvent(
                origin=SystemId("Worker"),
                origin_idx=worker_index,
                session=session_id,
                event=event,
            )
        )
        worker_index += 1

    async def node_info(info: MemoryUsage | NodeBackends) -> None:
        await worker_reports(
            NodeGatheredInfo(
                when=str(datetime.now(tz=timezone.utc)), node_id=node_id, info=info
            )
        )

    async def send(command: Command) -> None:
        await command_sender.send(
            ForwarderCommand(origin=SystemId("API"), command=command)
        )

    master = Master(
        node_id,
        session_id,
        event_sender=event_sender,
        global_event_sender=global_sender,
        local_event_receiver=local_receiver,
        command_receiver=command_receiver,
        download_command_sender=download_sender,
    )
    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        tg.start_soon(event_router)
        memory = Memory.from_bytes(678948 * 1024)
        await node_info(
            MemoryUsage(
                ram_total=memory,
                ram_available=memory,
                swap_total=Memory.from_bytes(0),
                swap_available=Memory.from_bytes(0),
            )
        )
        await node_info(NodeBackends(backends=[Backend.MlxMetal]))
        await _until(lambda: bool(master.state.node_backends))

        deployment = Deployment(
            deployment_id=DeploymentId(),
            model_card=MODEL,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        )
        await send(CreateDeployment(command_id=CommandId(), deployment=deployment))

        # The keeper places an instance, and the deployment names it
        await _until(lambda: bool(master.state.instances))
        (first,) = master.state.instances
        await _until(
            lambda: master.state.deployments[deployment.deployment_id].instance_id
            == first,
        )

        # Its runner loads, and the keeper sees it ready
        (runner_id,) = master.state.instances[
            first
        ].shard_assignments.node_to_runner.values()
        await worker_reports(
            RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerReady())
        )
        await _until(lambda: runner_id in master.state.runners)
        await anyio.sleep(1.5)

        # Someone deletes the instance: the keeper places another at once
        await send(DeleteInstance(command_id=CommandId(), instance_id=first))
        await _until(
            lambda: bool(master.state.instances)
            and first not in master.state.instances,
        )
        assert len(master.state.instances) == 1

        # Deleting the deployment deletes its instance, and nothing is placed again
        await send(
            DeleteDeployment(
                command_id=CommandId(), deployment_id=deployment.deployment_id
            )
        )
        await _until(
            lambda: not master.state.deployments and not master.state.instances,
        )
        await anyio.sleep(2.5)
        assert master.state.instances == {}

        event_sender.close()
        await master.shutdown()
