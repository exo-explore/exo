"""A master to drive in tests, with ready instances on a node it has just heard from."""

from datetime import datetime, timezone

import anyio

from exo.master.deployments import spec_of
from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    AddCustomModelCard,
    Command,
    CreateDeployment,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.common import DeploymentId, NodeId, SessionId, SystemId
from exo.shared.types.deployments import Deployment
from exo.shared.types.events import (
    CustomModelCardAdded,
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    StateSnapshot,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import Instance, InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, RunnerReady, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import Receiver, channel

MODEL = ModelId("test-model")
NODE = NodeId("node")
LOST_NODE = NodeId("lost-node")


def _card(model_id: ModelId = MODEL) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def instance_on(node: NodeId) -> MlxRingInstance:
    runner = RunnerId()
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=MODEL,
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=_card(),
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                )
            },
            node_to_runner={node: runner},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def deployment_of(instance: Instance) -> Deployment:
    """The deployment an instance runs, as the master records it when it places one."""
    shard = next(iter(instance.shard_assignments.runner_to_shard.values()))
    return Deployment(
        deployment_id=instance.deployment_id,
        spec=spec_of(instance),
        model_card=shard.model_card,
    )


def launch_of(instance: MlxRingInstance) -> CreateDeployment:
    """A launch of the placement, as the API sends it"""
    deployment_id = DeploymentId()
    instance = instance.model_copy(update={"deployment_id": deployment_id})
    shard = next(iter(instance.shard_assignments.runner_to_shard.values()))
    return CreateDeployment(
        deployment_id=deployment_id,
        spec=spec_of(instance),
        model_card=shard.model_card,
        instance=instance,
    )


def master_with(
    *instances: MlxRingInstance,
) -> tuple[Master, Receiver[GlobalForwarderEvent]]:
    """A master with the instances, ready, on a node it has just heard from, and what it
    sends to every node."""
    node_id = get_node_zid()
    global_event_sender, broadcast = channel[GlobalForwarderEvent]()
    topology = Topology()
    topology.add_node(NODE)
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        command_receiver=channel[ForwarderCommand]()[1],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=global_event_sender,
        snapshot_sender=channel[StateSnapshot]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        membership=FakeMembership(node_id, NODE),
        initial_state=State(
            topology=topology,
            last_seen={NODE: datetime.now(tz=timezone.utc)},
            instances={instance.instance_id: instance for instance in instances},
            deployments={
                instance.deployment_id: deployment_of(instance)
                for instance in instances
            },
            runners={
                runner: RunnerReady()
                for instance in instances
                for runner in instance.shard_assignments.runner_to_shard
            },
        ),
    )
    return master, broadcast


def marker() -> AddCustomModelCard:
    """A command sent after others: once its event is out, the master has processed them"""
    return AddCustomModelCard(model_card=_card(ModelId("marker")))


async def process(
    master: Master,
    broadcast: Receiver[GlobalForwarderEvent],
    *sent: Command,
) -> list[Event]:
    """Have the master process the commands back to back, then a marker, and return what it
    sent before the marker's event."""
    commands_sender, commands = channel[ForwarderCommand]()
    master.command_receiver = commands
    output: list[Event] = []
    async with anyio.create_task_group() as tg:
        tg.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        for command in (*sent, marker()):
            await commands_sender.send(
                ForwarderCommand(origin=SystemId(), command=command)
            )
        with anyio.fail_after(5):
            while not any(isinstance(event, CustomModelCardAdded) for event in output):
                output.append((await broadcast.receive()).event)
        tg.cancel_scope.cancel()
    return output[:-1]
