"""Deployments: the models users launched, which the cluster keeps running.

A deployment records what the user asked for (its spec). The master places it as an instance,
and when the instance is lost (a node left) it places it again: with the same model and the
user's constraints, on nodes that already have the model's files. Only a user's stop removes a
deployment. What placement chose before (`last_nodes`) is a preference for the next placement,
never a constraint, so a model can come back on fewer or other nodes when the user didn't fix
them.
"""

from datetime import timedelta
from typing import Literal

from pydantic import Field

from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.common import DeploymentId, NodeId
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.pydantic_ext import FrozenModel

# A deployment that loses a device this many times within CRASH_LOOP_WINDOW is paused instead of
# placed again. Each such loss is only acted on once the device has been gone for the membership's
# grace period (exo.routing.membership.GRACE, 30 s) and costs a reload, so 3 in 10 minutes means
# the model was down for much of that time; a Mac that sleeps twice in that time doesn't reach it.
CRASH_LOOP_LIMIT = 3
CRASH_LOOP_WINDOW = timedelta(minutes=10)


class DeploymentSpec(FrozenModel):
    """What the user asked for. Only these constrain where the model is placed again."""

    model_id: ModelId
    sharding: Sharding
    instance_meta: InstanceMeta
    # The fewest nodes the user wants it on; 1 lets it come back on a single node that fits
    min_nodes: int = Field(default=1, ge=1)
    # Nodes it must always include (the dashboard's node filter)
    pinned_nodes: list[NodeId] = []


# A deployment that has had no instance for this long is held until the user resumes it,
# rather than load by itself when it fits at last. Long enough to cover the absences after
# which a user expects the model back by itself (a Mac asleep through a meeting or a lunch
# break, an update with a restart); short enough that a model never loads hours after the user
# has moved on. A product choice, not a measurement.
WAIT_LIMIT = timedelta(hours=1)

# Why a deployment isn't placed again until the user resumes it:
# CrashLoop: it lost a device CRASH_LOOP_LIMIT times within CRASH_LOOP_WINDOW.
# StartFailed: its runner kept failing to start.
# WaitedTooLong: it had no instance for WAIT_LIMIT.
# Restored: the whole cluster restarted, and no node had it running since: models aren't
# loaded again by themselves when the Macs start up.
HoldKind = Literal["CrashLoop", "StartFailed", "WaitedTooLong", "Restored"]

# Stopped deployments remembered (their tombstones), so that a node that was away while a
# deployment was stopped doesn't bring it back when it rejoins. A tombstone only matters to a
# node that was away during the stop; this many cover months of use, at about 15 KB.
REMOVED_DEPLOYMENTS_KEPT = 200


class Hold(FrozenModel):
    kind: HoldKind
    message: str | None = None


# Why a deployment can't be placed yet. Coarse on purpose: it changes only when the situation
# does, not each time a node's free memory moves, and each change is an event every node gets.
# DeviceAway: a device it is pinned to isn't here, or (with `until`) it waits for the devices it
# last ran on: a launch the master learned of from a node, or a model whose device quit.
# NoDevicesWithFiles / NotEnoughDevices: too few connected devices have its files.
# DoesNotFit: the devices that have its files can't hold it now.
WaitKind = Literal["DeviceAway", "NoDevicesWithFiles", "NotEnoughDevices", "DoesNotFit"]


class Waiting(FrozenModel):
    kind: WaitKind
    needed: int | None = None
    have: int | None = None
    # An ISO 8601 time: a datetime field in an event can't be read back from JSON
    until: str | None = None


class DeploymentRecord(FrozenModel):
    """What every node keeps of a deployment, on disk and when it tells a new master: what the
    user asked for, the revision of their latest change, and the nodes it last ran on, so that
    a launch a merge keeps comes back where it ran. Holds and waits aren't kept: they are a
    master's own decisions, from what it sees, and never travel to another master."""

    deployment_id: DeploymentId
    spec: DeploymentSpec
    model_card: ModelCard
    revision: int = 0
    last_nodes: list[NodeId] = []


class Deployment(FrozenModel):
    deployment_id: DeploymentId
    spec: DeploymentSpec
    # The model's card when it was launched: placing it again needs nothing else
    model_card: ModelCard
    # Counts the user's changes after the launch (a resume): of two copies of a deployment, the
    # one with the user's latest change wins. The master's own decisions (holds, waits) are
    # derived from what it sees, and never count
    revision: int = 0
    hold: Hold | None = None
    # Set by the master as instances come and go
    # The nodes its last instance ran on: preferred when it is placed again
    last_nodes: list[NodeId] = []
    # Not placed again before this ISO time unless its nodes are back: a launch the master
    # learned of from a node waits for the nodes it last ran on (MERGED_LAUNCH_WAIT), and a
    # model whose node quit waits for it (LEAVE_GRACE)
    not_before: str | None = None
    # When it lost its instance (ISO); none while it has one
    lost_at: str | None = None
    # ISO times it lost a device, within CRASH_LOOP_WINDOW of the latest
    losses: list[str] = []
    waiting: Waiting | None = None
    # Something to know about how it runs, e.g. that it fell back from RDMA to TCP
    note: str | None = None
