# pyright: reportPrivateUsage=false
"""A running master keeps the models users launched running: a model whose instance lost a node
is placed again by itself, with the user's constraints, on nodes that have its files.

Before, a lost model was gone until the user launched it again: 48-61 manual relaunches in 2 h of
churn on a 5-node cluster.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import anyio
import pytest

import exo.master.main as master_main
from exo.master.deployments import spec_of
from exo.master.main import Master
from exo.master.tests.conftest import create_node_network, create_socket_connection
from exo.routing.membership import Quorum
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.apply import event_apply
from exo.shared.deployment_store import record_of
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    AnnounceDeployments,
    Command,
    CommandId,
    CreateDeployment,
    DeleteDeployment,
    ForwarderCommand,
    ForwarderDownloadCommand,
    InstanceBroken,
    ResumeDeployment,
)
from exo.shared.types.common import DeploymentId, NodeId, SessionId, SystemId
from exo.shared.types.deployments import (
    Deployment,
    DeploymentRecord,
    DeploymentSpec,
)
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    NodeDownloadProgress,
    NodeGatheredInfo,
    StateSnapshot,
    TopologyEdgeCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.topology import Connection
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import InstanceId, InstanceMeta, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding
from exo.utils.channels import Sender, channel
from exo.utils.info_gatherer.info_gatherer import NodeBackends, NodeNetworkInterfaces

MODEL = ModelCard(
    model_id=ModelId("test/deployed-model"),
    storage_size=Memory.from_mb(100),
    n_layers=8,
    hidden_size=64,
    supports_tensor=False,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)
STAYS, LEAVES, JOINS = NodeId("stays"), NodeId("leaves"), NodeId("joins")
# In the cluster but without the model's files: losing one node of three is not a split
BYSTANDER = NodeId("bystander")
MASTER_NODE = NodeId("master-node")
SESSION = SessionId(master_node_id=MASTER_NODE, election_clock=0)


def instance_on(*nodes: NodeId) -> MlxRingInstance:
    runners = {node: RunnerId() for node in nodes}
    layers = MODEL.n_layers // len(nodes)
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=MODEL.model_id,
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=MODEL,
                    device_rank=rank,
                    world_size=len(nodes),
                    start_layer=rank * layers,
                    end_layer=(rank + 1) * layers,
                    n_layers=MODEL.n_layers,
                )
                for rank, runner in enumerate(runners.values())
            },
            node_to_runner=runners,
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def launch_on(*nodes: NodeId) -> CreateDeployment:
    """A launch of MODEL on these nodes, as the API sends one."""
    deployment_id = DeploymentId()
    return CreateDeployment(
        deployment_id=deployment_id,
        spec=DeploymentSpec(
            model_id=MODEL.model_id,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
        ),
        model_card=MODEL,
        instance=instance_on(*nodes).model_copy(
            update={"deployment_id": deployment_id}
        ),
    )


def node_info(node_id: NodeId, *, has_files: bool = True) -> list[Event]:
    """What a node reports when it joins: memory, backends, addresses and, if it has them,
    the model's files."""
    when = str(datetime.now(tz=timezone.utc))
    events: list[Event] = [
        NodeGatheredInfo(
            node_id=node_id,
            when=when,
            info=MemoryUsage.from_bytes(
                ram_total=64 * 1024**3,
                ram_available=32 * 1024**3,
                swap_total=0,
                swap_available=0,
            ),
        ),
        NodeGatheredInfo(
            node_id=node_id, when=when, info=NodeBackends(backends=[Backend.MlxMetal])
        ),
        NodeGatheredInfo(
            node_id=node_id,
            when=when,
            info=NodeNetworkInterfaces(ifaces=list(create_node_network().interfaces)),
        ),
    ]
    if has_files:
        shard = next(
            iter(instance_on(node_id).shard_assignments.runner_to_shard.values())
        )
        events.append(
            NodeDownloadProgress(
                download_progress=DownloadCompleted(
                    node_id=node_id, shard_metadata=shard, total=MODEL.storage_size
                )
            )
        )
    return events


def edges_between(a: NodeId, b: NodeId) -> list[Event]:
    return [
        TopologyEdgeCreated(
            conn=Connection(source=a, sink=b, edge=create_socket_connection(1))
        ),
        TopologyEdgeCreated(
            conn=Connection(source=b, sink=a, edge=create_socket_connection(2))
        ),
    ]


def deployed(instance: MlxRingInstance, min_nodes: int = 1) -> State:
    """A deployment of MODEL, running as `instance`."""
    deployment_id = DeploymentId()
    instance = instance.model_copy(update={"deployment_id": deployment_id})
    return State(
        instances={instance.instance_id: instance},
        deployments={
            deployment_id: Deployment(
                deployment_id=deployment_id,
                spec=spec_of(instance, min_nodes=min_nodes),
                model_card=MODEL,
            )
        },
    )


def cluster(
    state: State, *nodes: NodeId, without_files: tuple[NodeId, ...] = ()
) -> State:
    """`state` on a cluster of `nodes`, all connected to each other."""
    for node_id in nodes:
        for event in node_info(node_id, has_files=node_id not in without_files):
            state = event_apply(event, state)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            for event in edges_between(a, b):
                state = event_apply(event, state)
    return state


@dataclass
class Cluster:
    master: Master
    # The master's membership view: every node of the initial cluster is Alive in it
    membership: FakeMembership
    local: Sender[LocalForwarderEvent]
    commands: Sender[ForwarderCommand]
    next_idx: int = 0

    async def send(self, events: list[Event]) -> None:
        """Report events from the nodes, as their workers would."""
        for event in events:
            idx, self.next_idx = self.next_idx, self.next_idx + 1
            await self.local.send(
                LocalForwarderEvent(
                    origin=SystemId("nodes"),
                    origin_idx=idx,
                    session=SESSION,
                    event=event,
                )
            )

    def lose(self, *nodes: NodeId, quorum: Quorum = "Majority") -> None:
        """The nodes are Gone in the master's membership view (they were Suspect for the grace
        period), with this quorum read"""
        self.membership.quorum = quorum
        for node in nodes:
            self.membership.set(node, "Gone")

    async def join(
        self, node: NodeId, *linked_to: NodeId, has_files: bool = True
    ) -> None:
        """The node is Alive (back, or new), and reports itself and its links"""
        self.membership.regain(node)
        await self.send(
            node_info(node, has_files=has_files)
            + [event for other in linked_to for event in edges_between(other, node)]
        )

    async def command(self, command: Command) -> None:
        await self.commands.send(
            ForwarderCommand(origin=SystemId("api"), command=command)
        )

    @property
    def state(self) -> State:
        return self.master.state

    def model_nodes(self) -> list[frozenset[NodeId]]:
        return [
            frozenset(instance.shard_assignments.node_to_runner)
            for instance in self.state.instances.values()
        ]

    def deployment(self) -> Deployment:
        (deployment,) = self.state.deployments.values()
        return deployment


@asynccontextmanager
async def running_master(initial: State) -> AsyncIterator[Cluster]:
    """A running master, and a way to send it what nodes report and commands."""
    global_send, _global_receive = channel[GlobalForwarderEvent]()
    command_send, command_receive = channel[ForwarderCommand]()
    local_send, local_receive = channel[LocalForwarderEvent]()
    download_send, _download_receive = channel[ForwarderDownloadCommand]()
    snapshot_send, _snapshot_receive = channel[StateSnapshot]()
    membership = FakeMembership(MASTER_NODE, *initial.topology.list_nodes())
    master = Master(
        MASTER_NODE,
        SESSION,
        global_event_sender=global_send,
        snapshot_sender=snapshot_send,
        local_event_receiver=local_receive,
        command_receiver=command_receive,
        download_command_sender=download_send,
        membership=membership,
        initial_state=initial,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        yield Cluster(
            master=master,
            membership=membership,
            local=local_send,
            commands=command_send,
        )
        tg.cancel_scope.cancel()


async def wait_until(condition: Callable[[], bool], seconds: float = 5) -> None:
    with anyio.fail_after(seconds):
        while not condition():
            await anyio.sleep(0.01)


@pytest.fixture(autouse=True)
def fast_master(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(master_main, "PLAN_INTERVAL", 0.05)


async def test_a_model_that_lost_a_node_comes_back_on_the_nodes_left() -> None:
    """The user didn't fix the node count: it comes back on one node, which holds it."""
    initial = cluster(
        deployed(instance_on(STAYS, LEAVES)),
        STAYS,
        LEAVES,
        BYSTANDER,
        without_files=(BYSTANDER,),
    )
    async with running_master(initial) as running:
        running.lose(LEAVES)

        await wait_until(lambda: running.model_nodes() == [frozenset({STAYS})])
        assert running.deployment().last_nodes == [STAYS, LEAVES]


async def test_a_model_the_user_put_on_two_nodes_waits_for_two_and_says_why() -> None:
    initial = cluster(
        deployed(instance_on(STAYS, LEAVES), min_nodes=2),
        STAYS,
        LEAVES,
        BYSTANDER,
        without_files=(BYSTANDER,),
    )
    async with running_master(initial) as running:
        running.lose(LEAVES)
        await wait_until(lambda: running.deployment().waiting is not None)
        waiting = running.deployment().waiting
        assert waiting is not None
        assert (waiting.kind, waiting.needed, waiting.have) == (
            "NotEnoughDevices",
            2,
            1,
        )

        # A node with the model's files joins
        await running.join(JOINS, STAYS, BYSTANDER)

        await wait_until(lambda: running.model_nodes() == [frozenset({STAYS, JOINS})])
        assert running.deployment().waiting is None


async def test_a_node_without_the_models_files_is_not_used() -> None:
    """Placing a model again never starts a download nobody asked for."""
    initial = cluster(
        deployed(instance_on(STAYS, LEAVES), min_nodes=2),
        STAYS,
        LEAVES,
        BYSTANDER,
        without_files=(BYSTANDER,),
    )
    async with running_master(initial) as running:
        running.lose(LEAVES)
        await wait_until(lambda: running.deployment().waiting is not None)

        # BYSTANDER has memory and is connected, but none of the model's files
        await anyio.sleep(0.5)
        assert running.model_nodes() == []


async def test_a_model_is_placed_again_only_on_nodes_this_node_reaches() -> None:
    """A node the master's membership has as Suspect is still in the cluster (it may be back
    within the grace period), but a model placed on it couldn't load."""
    initial = cluster(
        deployed(instance_on(LEAVES)),
        STAYS,
        LEAVES,
        JOINS,
    )
    async with running_master(initial) as running:
        running.membership.set(STAYS, "Suspect", "Disconnected")
        running.lose(LEAVES)

        await wait_until(lambda: running.model_nodes() == [frozenset({JOINS})])
        assert STAYS in running.state.topology.list_nodes()


async def test_stopping_a_waiting_model_ends_it() -> None:
    initial = cluster(
        deployed(instance_on(STAYS, LEAVES), min_nodes=2),
        STAYS,
        LEAVES,
        BYSTANDER,
        without_files=(BYSTANDER,),
    )
    async with running_master(initial) as running:
        running.lose(LEAVES)
        await wait_until(lambda: running.deployment().waiting is not None)

        await running.command(
            DeleteDeployment(deployment_id=running.deployment().deployment_id)
        )
        await wait_until(lambda: running.state.deployments == {})

        # A node with the files joins: nothing is started
        await running.join(JOINS, STAYS)
        await anyio.sleep(0.5)
        assert running.model_nodes() == []


async def test_a_model_the_user_stopped_is_not_placed_again() -> None:
    initial = cluster(deployed(instance_on(STAYS, LEAVES)), STAYS, LEAVES, BYSTANDER)
    async with running_master(initial) as running:
        await running.command(
            DeleteDeployment(deployment_id=running.deployment().deployment_id)
        )
        await wait_until(lambda: running.state.instances == {})
        await anyio.sleep(0.5)
        assert running.state.deployments == {}
        assert running.model_nodes() == []


async def test_a_new_launch_of_a_waiting_model_replaces_it() -> None:
    """Launched by hand while it waits (e.g. with the dashboard's "Start again"): the model
    isn't started a second time once the first one fits."""
    initial = cluster(
        deployed(instance_on(STAYS, LEAVES), min_nodes=2),
        STAYS,
        LEAVES,
        BYSTANDER,
        without_files=(BYSTANDER,),
    )
    async with running_master(initial) as running:
        running.lose(LEAVES)
        await wait_until(lambda: running.deployment().waiting is not None)
        waiting_id = running.deployment().deployment_id

        launch = launch_on(STAYS)
        by_hand = launch.deployment_id
        await running.command(launch)
        await wait_until(lambda: list(running.state.deployments) == [by_hand])
        assert waiting_id not in running.state.deployments

        # A node with the files joins: still one instance
        await running.join(JOINS, STAYS)
        await anyio.sleep(0.5)
        assert running.model_nodes() == [frozenset({STAYS})]


async def test_on_one_side_of_an_even_split_a_lost_model_is_kept_for_its_node() -> None:
    """Two Macs lose each other: the other one may be serving the model, so it isn't loaded
    here too. The master keeps the gone node listed, with its instance, for the split hold:
    the model counts as placed, and nothing is placed here. A node back within the hold has
    its model as it was. Once the hold is over (the quorum read is a majority of the nodes
    this one reaches), the node is removed and its model is placed here at once."""
    initial = cluster(deployed(instance_on(LEAVES)), STAYS, LEAVES)
    (kept,) = initial.instances
    async with running_master(initial) as running:
        running.lose(LEAVES, quorum="Half")
        # STAYS has the files and room: it is still not placed there
        await anyio.sleep(0.5)
        assert list(running.state.instances) == [kept]
        assert running.deployment().waiting is None

        # LEAVES is back within the split hold: nothing moved
        running.membership.regain(LEAVES)
        running.membership.quorum = "Majority"
        await anyio.sleep(0.3)
        assert list(running.state.instances) == [kept]

        # Lost again, and not back by the end of the split hold
        running.lose(LEAVES, quorum="Half")
        await anyio.sleep(0.3)
        assert list(running.state.instances) == [kept]
        running.lose(LEAVES, quorum="Majority")
        await wait_until(lambda: running.model_nodes() == [frozenset({STAYS})])
        assert LEAVES not in running.state.topology.list_nodes()


async def test_a_launch_onto_a_node_this_node_cant_reach_is_refused() -> None:
    """Its model couldn't load there. The API sends the launch again until it shows, by
    when the node may be back."""
    initial = cluster(State(), STAYS, LEAVES)
    async with running_master(initial) as running:
        running.membership.set(LEAVES, "Suspect", "Disconnected")
        await running.command(launch_on(LEAVES))
        await anyio.sleep(0.3)
        assert running.state.deployments == {}
        assert running.state.instances == {}


async def test_nothing_is_placed_while_the_masters_node_is_resuming() -> None:
    """The master's node was stopped (frozen, or its Mac asleep): its peers are still
    reconnecting, and a model placed now would leave out nodes that are about to be back."""
    initial = cluster(
        deployed(instance_on(STAYS, LEAVES)),
        STAYS,
        LEAVES,
        BYSTANDER,
        without_files=(BYSTANDER,),
    )
    async with running_master(initial) as running:
        running.membership.self_state = "Resuming"
        running.lose(LEAVES)
        await wait_until(lambda: running.state.instances == {})
        await anyio.sleep(0.3)
        assert running.model_nodes() == []

        running.membership.self_state = "Alive"
        await wait_until(lambda: running.model_nodes() == [frozenset({STAYS})])


async def test_a_model_that_failed_to_start_waits_for_the_user() -> None:
    initial = cluster(deployed(instance_on(STAYS)), STAYS, LEAVES)
    (instance,) = initial.instances.values()
    async with running_master(initial) as running:
        await running.command(
            InstanceBroken(
                instance_id=instance.instance_id,
                reason="StartFailed",
                message="its runner here failed to start 3 times",
            )
        )
        await wait_until(lambda: running.deployment().hold is not None)
        hold = running.deployment().hold
        assert hold is not None
        assert (hold.kind, hold.message) == (
            "StartFailed",
            "its runner here failed to start 3 times",
        )
        await anyio.sleep(0.5)
        assert running.model_nodes() == []

        await running.command(
            ResumeDeployment(
                deployment_id=running.deployment().deployment_id,
                revision=running.deployment().revision,
            )
        )
        await wait_until(lambda: running.model_nodes() != [])


async def test_a_late_report_on_a_lost_instance_does_not_stop_its_model() -> None:
    """A worker gives up on an instance after it was already lost and placed again: its
    report is about an instance that is gone, and changes nothing."""
    initial = cluster(
        deployed(instance_on(STAYS, LEAVES)),
        STAYS,
        LEAVES,
        BYSTANDER,
        without_files=(BYSTANDER,),
    )
    (lost,) = initial.instances.values()
    async with running_master(initial) as running:
        running.lose(LEAVES)
        await wait_until(lambda: running.model_nodes() == [frozenset({STAYS})])

        await running.command(
            InstanceBroken(instance_id=lost.instance_id, reason="StartFailed")
        )
        await anyio.sleep(0.5)
        assert running.deployment().hold is None
        assert running.model_nodes() == [frozenset({STAYS})]


async def test_a_model_that_keeps_losing_its_device_is_paused() -> None:
    """Placed again on a node that keeps dropping out, it is paused after CRASH_LOOP_LIMIT
    losses, instead of being loaded again and again."""
    initial = cluster(
        deployed(instance_on(LEAVES), min_nodes=1),
        LEAVES,
        STAYS,
        BYSTANDER,
        without_files=(STAYS, BYSTANDER),
    )
    async with running_master(initial) as running:
        for _ in range(3):
            # LEAVES drops out; the model is lost, and comes back when LEAVES does
            running.lose(LEAVES)
            await wait_until(lambda: running.model_nodes() == [])
            if running.deployment().hold is not None:
                break
            await running.join(LEAVES, STAYS, BYSTANDER)
            await wait_until(
                lambda: running.model_nodes() == [frozenset({LEAVES})]
                or running.deployment().hold is not None
            )
        await wait_until(lambda: running.deployment().hold is not None)
        hold = running.deployment().hold
        assert hold is not None and hold.kind == "CrashLoop"
        assert len(running.deployment().losses) == 3


async def test_a_launch_by_id_is_launched_once() -> None:
    """A command sent again (by the API, while the master changed) doesn't start a second
    copy: the deployment id is the same."""
    initial = cluster(State(), STAYS, LEAVES, BYSTANDER)
    launch = launch_on(STAYS).model_copy(update={"command_id": CommandId()})
    async with running_master(initial) as running:
        await running.command(launch)
        await running.command(launch.model_copy(update={"command_id": CommandId()}))
        await wait_until(lambda: running.model_nodes() != [])
        await anyio.sleep(0.3)
        assert list(running.state.deployments) == [launch.deployment_id]
        assert len(running.state.instances) == 1


async def test_a_model_stopped_right_after_its_launch_does_not_start() -> None:
    """The stop reaches the master before its own launch events come back to it."""
    initial = cluster(State(), STAYS, LEAVES, BYSTANDER)
    launch = launch_on(STAYS)
    async with running_master(initial) as running:
        await running.command(launch)
        await running.command(DeleteDeployment(deployment_id=launch.deployment_id))
        await anyio.sleep(0.5)
        assert running.state.deployments == {}
        assert running.state.instances == {}


def _launched_elsewhere(nodes: State) -> tuple[DeploymentRecord, State]:
    """A deployment some node knows of that this master's state lacks."""
    other = deployed(instance_on(STAYS))
    (deployment,) = other.deployments.values()
    return record_of(deployment), nodes


async def test_a_launch_a_new_master_never_saw_is_placed_when_a_node_tells_it() -> None:
    """The user launched on the side that lost the election (or the master that took over
    was behind): the node that has it tells the new master, which places it."""
    record, initial = _launched_elsewhere(cluster(State(), STAYS, LEAVES, BYSTANDER))
    async with running_master(initial) as running:
        await running.command(
            AnnounceDeployments(deployments=[record], removed={}, restored=[])
        )

        await wait_until(lambda: running.model_nodes() != [])
        assert list(running.state.deployments) == [record.deployment_id]


async def test_a_launch_a_new_master_never_saw_comes_back_on_its_own_node() -> None:
    """It ran on LEAVES, on the side that lost the election. After the heal LEAVES reports
    its files a moment later than STAYS, which has them too and more room: the launch waits
    for LEAVES rather than moving (on the other Mac in 1 of 2 runs)."""
    (deployment,) = deployed(instance_on(LEAVES)).deployments.values()
    record = record_of(deployment, [LEAVES])
    initial = cluster(State(), STAYS, LEAVES, BYSTANDER, without_files=(LEAVES,))
    async with running_master(initial) as running:
        await running.command(
            AnnounceDeployments(deployments=[record], removed={}, restored=[])
        )
        await anyio.sleep(0.5)
        assert running.model_nodes() == []
        # LEAVES says it has the files
        await running.send(node_info(LEAVES))

        await wait_until(lambda: running.model_nodes() != [])
        assert running.model_nodes() == [frozenset({LEAVES})]


async def test_a_stop_that_reaches_a_master_that_never_saw_the_launch_holds() -> None:
    """The user stops a model just as the master changes, and the new
    master is behind: it never saw the launch. The stop records a tombstone all the same, so
    a node that has the deployment can't bring it back. It recorded nothing, and the API
    counted the stop applied since the deployment wasn't there: the model came back with the
    next announcement."""
    launched = cluster(deployed(instance_on(STAYS)), STAYS, LEAVES)
    (deployment,) = launched.deployments.values()
    behind = launched.model_copy(update={"instances": {}, "deployments": {}})
    async with running_master(behind) as running:
        await running.command(DeleteDeployment(deployment_id=deployment.deployment_id))
        await wait_until(
            lambda: deployment.deployment_id in running.state.removed_deployments
        )

        await running.command(
            AnnounceDeployments(
                deployments=[record_of(deployment, [STAYS])],
                removed={},
                restored=[],
                running=[deployment.deployment_id],
            )
        )
        await anyio.sleep(0.5)
    assert running.state.deployments == {}
    assert running.model_nodes() == []


async def test_a_late_copy_of_a_stopped_launch_changes_nothing() -> None:
    """A launch the user stopped since arrives again (sent again, to a
    new master). It isn't launched, and doesn't replace the model's deployment that waits,
    as a new launch of the model would."""
    waiting_state = cluster(
        deployed(instance_on(LEAVES)), STAYS, LEAVES, without_files=(STAYS, LEAVES)
    )
    (waiting,) = waiting_state.deployments
    stopped = DeploymentId()
    state = waiting_state.model_copy(
        update={
            "instances": {},
            "removed_deployments": {stopped: "2026-11-03T12:00:00+00:00"},
        }
    )
    (deployment,) = waiting_state.deployments.values()
    async with running_master(state) as running:
        await running.command(
            CreateDeployment(
                deployment_id=stopped,
                spec=deployment.spec,
                model_card=MODEL,
                instance=instance_on(STAYS).model_copy(
                    update={"deployment_id": stopped}
                ),
            )
        )
        await anyio.sleep(0.5)
    assert list(running.state.deployments) == [waiting]
    assert waiting not in running.state.removed_deployments
    assert running.model_nodes() == []


async def test_a_stop_made_on_the_other_side_ends_the_model_here() -> None:
    initial = cluster(deployed(instance_on(STAYS)), STAYS, LEAVES, BYSTANDER)
    async with running_master(initial) as running:
        stopped = running.deployment()

        await running.command(
            AnnounceDeployments(
                deployments=[record_of(stopped)],
                removed={stopped.deployment_id: "2026-11-03T12:00:00+00:00"},
                restored=[],
            )
        )

        await wait_until(lambda: running.state.instances == {})
        assert running.state.deployments == {}
        assert stopped.deployment_id in running.state.removed_deployments


async def test_after_a_full_restart_models_wait_for_the_user() -> None:
    """Every node read it from its disk: it comes back held, and loads when resumed."""
    record, initial = _launched_elsewhere(cluster(State(), STAYS, LEAVES, BYSTANDER))
    async with running_master(initial) as running:
        await running.command(
            AnnounceDeployments(
                deployments=[record], removed={}, restored=[record.deployment_id]
            )
        )
        await wait_until(lambda: running.state.deployments != {})
        hold = running.deployment().hold
        assert hold is not None and hold.kind == "Restored"
        await anyio.sleep(0.5)
        assert running.model_nodes() == []

        await running.command(
            ResumeDeployment(
                deployment_id=record.deployment_id,
                revision=running.deployment().revision,
            )
        )
        await wait_until(lambda: running.model_nodes() != [])
