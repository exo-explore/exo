"""Tests for asymmetric drafter placement (Layer B).

When a model card declares ``drafter_eligible_nodes`` AND the cluster
has at least one such node alive, reachable from every target rank, and
with sufficient memory, placement appends a *drafter rank* to the
parent ``mx.distributed`` group on a separate node. Target ranks split
off into a target subgroup at runtime; the parent group is reserved for
``RemoteTransport`` send/recv between target rank 0 and the drafter
rank.

Coverage:
- Asymmetric placement is constructed when an eligible node is reachable
  with both backends (``MlxRing`` over socket, ``MlxJaccl`` over RDMA).
- Placement degrades loudly when no eligible node is alive, when every
  eligible node is already a target rank, or when the only eligible
  candidate has no reachable transport. The user's request still
  completes (placement returns *something*), and a
  ``DrafterPlacementDegraded`` event is emitted with the reason.
- Empty ``drafter_eligible_nodes`` preserves legacy behaviour.
- The drafter rank is always the LAST rank in the parent group.
"""

from collections.abc import Iterator

import pytest
from loguru import logger as loguru_logger

from exo.master.placement import place_instance
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_rdma_connection,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    DrafterPlacementDegradationReason,
    DrafterPlacementDegraded,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import NodeRdmaCtlStatus
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import (
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.shards import Sharding


@pytest.fixture
def loguru_capture() -> Iterator[list[str]]:
    captured: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: captured.append(str(message)), level="ERROR"
    )
    try:
        yield captured
    finally:
        loguru_logger.remove(sink_id)


def _drafter_aware_card(
    *,
    storage_bytes: int,
    eligible_nodes: list[NodeId],
    family: str = "gemma",
    base_model: str = "Gemma 4 31B",
    model_id: str = "mlx-community/gemma-4-31b-it-8bit",
) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_bytes(storage_bytes),
        n_layers=60,
        hidden_size=5376,
        num_key_value_heads=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family=family,
        base_model=base_model,
        drafter_model_ids=[
            ModelId("mlx-community/gemma-4-e2b-it-8bit"),
            ModelId("mlx-community/gemma-4-e4b-it-8bit"),
        ],
        drafter_eligible_nodes=eligible_nodes,
    )


def _bidi_socket(topology: Topology, a: NodeId, b: NodeId, ip: int) -> None:
    topology.add_connection(
        Connection(source=a, sink=b, edge=create_socket_connection(ip))
    )
    topology.add_connection(
        Connection(source=b, sink=a, edge=create_socket_connection(ip + 1))
    )


def _bidi_rdma(topology: Topology, a: NodeId, b: NodeId, iface: int) -> None:
    topology.add_connection(
        Connection(source=a, sink=b, edge=create_rdma_connection(iface))
    )
    topology.add_connection(
        Connection(source=b, sink=a, edge=create_rdma_connection(iface + 1))
    )


def test_asymmetric_single_node_target_stays_on_ring() -> None:
    """Single-node target + RDMA-reachable drafter => asymmetric ring.

    Codex P1.4 (PR #20): the V3+ wire keeps the drafter rank OUT of
    ``mx.distributed`` -- it talks to target rank 0 over a plain TCP
    socket. A single-rank target therefore never needs ``Group.split``
    / ``send/recv`` and stays on ``MlxRing`` even when an asymmetric
    drafter is reachable. Pre-fix the placement auto-upgraded
    ``MlxRing -> MlxJaccl`` here, which then triggered the JACCL
    Thunderbolt-IPv4 preflight on Wi-Fi/Ethernet single-node deploys
    and caused unnecessary placement failures.
    """
    target_node, drafter_node = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(drafter_node)
    _bidi_socket(topology, target_node, drafter_node, ip=2)
    _bidi_rdma(topology, target_node, drafter_node, iface=4)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_node: create_node_network(),
            drafter_node: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    assert not degradations
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert instance.drafter_placement is not None
    placement = instance.drafter_placement
    assert placement.drafter_node_id == drafter_node
    assert placement.drafter_model_id == ModelId("mlx-community/gemma-4-e2b-it-8bit")
    assert placement.drafter_rank == 1  # target=1 rank, drafter is last (rank 1)
    # v3+ wire: drafter does not join mx.distributed -> parent_group_size
    # is the target-only rank count.
    assert instance.parent_group_size == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


def test_asymmetric_ring_socket_only_places_drafter_over_socket() -> None:
    """Single-node ring target + socket-only drafter places drafter over TCP.

    v3+ wire decoupled the drafter from ``mx.distributed`` -- the wire
    runs over a plain TCP socket. RDMA is therefore no longer required
    for asymmetric placement; a socket-only path between target rank 0
    and the drafter node is sufficient. Codex P1.4: single-node
    targets stay on ``MlxRing`` (no ``MlxJaccl`` auto-upgrade) and
    the drafter wire still runs over TCP regardless of the target
    backend.
    """
    target_node, drafter_node = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(drafter_node)
    _bidi_socket(topology, target_node, drafter_node, ip=2)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_node: create_node_network(),
            drafter_node: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert instance.drafter_placement is not None
    assert instance.drafter_placement.drafter_node_id == drafter_node
    # Target stays single-rank; drafter rides TCP regardless.
    assert instance.parent_group_size == 1
    assert not degradations


def test_asymmetric_jaccl_places_drafter_with_rdma_reachability() -> None:
    """Two-node target (RDMA cycle) + RDMA-reachable drafter => asymmetric jaccl.

    Single-node targets always land on ``MlxRing`` (Codex P1.4: the
    drafter wire is a TCP socket independent of ``mx.distributed``,
    so single-rank cycles never need jaccl). To exercise asymmetric
    jaccl we therefore need the target to span 2 RDMA-connected nodes
    plus a 3rd drafter node with RDMA edges to both.
    """
    target_a, target_b, drafter_node = NodeId(), NodeId(), NodeId()
    topology = Topology()
    for n in (target_a, target_b, drafter_node):
        topology.add_node(n)
    # Target cycle has bidirectional RDMA between target_a and target_b
    _bidi_rdma(topology, target_a, target_b, iface=10)
    _bidi_socket(topology, target_a, target_b, ip=12)
    # Drafter has bidirectional RDMA + socket to both target ranks.
    _bidi_rdma(topology, target_a, drafter_node, iface=20)
    _bidi_rdma(topology, target_b, drafter_node, iface=22)
    _bidi_socket(topology, target_a, drafter_node, ip=14)
    _bidi_socket(topology, target_b, drafter_node, ip=16)

    # Use a Qwen-family card so the test isn't subject to Gemma 4's
    # "no multi-node Pipeline" restriction. Tensor sharding works across
    # 2 RDMA-connected nodes when hidden_size is divisible by world_size.
    card = _drafter_aware_card(
        storage_bytes=40_000_000_000,
        eligible_nodes=[drafter_node],
        family="qwen",
        base_model="Qwen3 30B",
        model_id="mlx-community/Qwen3-30B-A3B-4bit",
    )
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=card,
        # min_nodes=2 forces multi-node target so the placement layer
        # keeps MlxJaccl instead of rewriting to MlxRing.
        min_nodes=2,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_a: create_node_memory(32_000_000_000),
            target_b: create_node_memory(32_000_000_000),
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_a: create_node_network(),
            target_b: create_node_network(),
            drafter_node: create_node_network(),
        },
        node_rdma_ctl={
            target_a: NodeRdmaCtlStatus(enabled=True),
            target_b: NodeRdmaCtlStatus(enabled=True),
            drafter_node: NodeRdmaCtlStatus(enabled=True),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    assert not degradations, [(e.reason, e.detail) for e in degradations]
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance)
    assert instance.drafter_placement is not None
    placement = instance.drafter_placement
    assert placement.drafter_node_id == drafter_node
    assert placement.drafter_rank == 2  # logical telemetry index past target ranks
    # v3+ wire: drafter is on a TCP socket, not in mx.distributed.
    # parent_group_size and jaccl_devices cover only the 2 target ranks.
    assert instance.parent_group_size == 2
    assert len(instance.jaccl_devices) == 2
    assert len(instance.jaccl_devices[0]) == 2
    # Drafter node does not coordinate the target's mx.distributed group.
    assert drafter_node not in instance.jaccl_coordinators


def test_asymmetric_jaccl_socket_only_drafter_succeeds(
    loguru_capture: list[str],
) -> None:
    """Two-node jaccl target + socket-only drafter places successfully.

    v3+ wire: drafter IPC runs over a plain TCP socket independent of
    the target's ``mx.distributed`` group. So a socket-only path from
    target rank 0 to the drafter node is sufficient even when the
    target ranks themselves are coordinating over jaccl/RDMA. No
    degradation event should fire.
    """
    target_a, target_b, drafter_node = NodeId(), NodeId(), NodeId()
    topology = Topology()
    for n in (target_a, target_b, drafter_node):
        topology.add_node(n)
    # Target cycle has bidirectional RDMA; drafter only has socket edges.
    _bidi_rdma(topology, target_a, target_b, iface=30)
    _bidi_socket(topology, target_a, target_b, ip=32)
    _bidi_socket(topology, target_a, drafter_node, ip=34)
    _bidi_socket(topology, target_b, drafter_node, ip=36)

    card = _drafter_aware_card(
        storage_bytes=40_000_000_000,
        eligible_nodes=[drafter_node],
        family="qwen",
        base_model="Qwen3 30B",
        model_id="mlx-community/Qwen3-30B-A3B-4bit",
    )
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=card,
        min_nodes=2,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_a: create_node_memory(32_000_000_000),
            target_b: create_node_memory(32_000_000_000),
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_a: create_node_network(),
            target_b: create_node_network(),
            drafter_node: create_node_network(),
        },
        node_rdma_ctl={
            target_a: NodeRdmaCtlStatus(enabled=True),
            target_b: NodeRdmaCtlStatus(enabled=True),
            drafter_node: NodeRdmaCtlStatus(enabled=True),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance)
    assert instance.drafter_placement is not None
    assert instance.drafter_placement.drafter_node_id == drafter_node
    # 2 target ranks + drafter on socket; mx.distributed is target-only.
    assert instance.parent_group_size == 2
    assert not degradations
    # No degradation log line either.
    joined = "\n".join(loguru_capture)
    assert "Drafter placement degraded" not in joined


def test_asymmetric_degrades_when_eligible_node_missing_from_topology(
    loguru_capture: list[str],
) -> None:
    """Eligible node id refers to a node not present in topology."""
    target_node = NodeId()
    missing_drafter_node = NodeId()  # Never added to topology.
    topology = Topology()
    topology.add_node(target_node)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[missing_drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {target_node: create_node_memory(64_000_000_000)},
        {target_node: create_node_network()},
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert instance.drafter_placement is None
    assert len(degradations) == 1
    assert (
        degradations[0].reason
        == DrafterPlacementDegradationReason.NoEligibleNodeAvailable
    )
    assert degradations[0].fallback == "single_device_drafter"
    joined = "\n".join(loguru_capture).lower()
    assert "drafter placement degraded" in joined


def test_asymmetric_degrades_when_eligible_node_in_target_cycle(
    loguru_capture: list[str],
) -> None:
    """Listing the target node itself as eligible is a misconfig => degrade."""
    target_node = NodeId()
    topology = Topology()
    topology.add_node(target_node)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[target_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {target_node: create_node_memory(64_000_000_000)},
        {target_node: create_node_network()},
        on_drafter_placement_degraded=degradations.append,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert instance.drafter_placement is None
    assert len(degradations) == 1
    assert (
        degradations[0].reason
        == DrafterPlacementDegradationReason.AllEligibleNodesInTargetCycle
    )
    del loguru_capture  # captured but content irrelevant beyond emission


def test_asymmetric_degrades_when_drafter_node_lacks_memory() -> None:
    """Drafter node reachable but below memory floor (~6GB) => degrade.

    RDMA-reachable so jaccl auto-upgrade is viable, but memory check
    rejects the candidate. Single-node target therefore reverts to
    symmetric MlxRing without drafter.
    """
    target_node, drafter_node = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(drafter_node)
    _bidi_socket(topology, target_node, drafter_node, ip=8)
    _bidi_rdma(topology, target_node, drafter_node, iface=40)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            drafter_node: create_node_memory(2_000_000_000),  # 2GB is below floor
        },
        {
            target_node: create_node_network(),
            drafter_node: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert instance.drafter_placement is None
    assert len(degradations) == 1
    assert (
        degradations[0].reason
        == DrafterPlacementDegradationReason.InsufficientDrafterMemory
    )


def test_empty_drafter_eligible_nodes_preserves_legacy_behaviour() -> None:
    """No eligible list => no asymmetric attempt, no degradation events."""
    target_node = NodeId()
    topology = Topology()
    topology.add_node(target_node)

    card = ModelCard(
        model_id=ModelId("mlx-community/gemma-4-31b-it-8bit"),
        storage_size=Memory.from_bytes(20_000_000_000),
        n_layers=60,
        hidden_size=5376,
        num_key_value_heads=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="gemma",
        base_model="Gemma 4 31B",
        drafter_model_ids=[ModelId("mlx-community/gemma-4-e2b-it-8bit")],
        drafter_eligible_nodes=[],
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {target_node: create_node_memory(64_000_000_000)},
        {target_node: create_node_network()},
        on_drafter_placement_degraded=degradations.append,
    )

    instance = next(iter(placements.values()))
    assert instance.drafter_placement is None
    assert not degradations  # no asymmetric attempt was made


def test_asymmetric_with_multiple_eligible_nodes_picks_first_reachable() -> None:
    """When multiple eligible nodes are listed, placement picks the first
    reachable (in card order). Earlier candidates that fail reachability
    are skipped silently (the search is best-effort, not first-fail).

    Single-node target auto-upgrades to jaccl, so the reachable drafter
    needs an RDMA edge (not just a socket edge); the unreachable drafter
    has no edges at all.
    """
    target_node = NodeId()
    unreachable_drafter = NodeId()
    reachable_drafter = NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(unreachable_drafter)
    topology.add_node(reachable_drafter)
    # Only reachable_drafter has socket + RDMA edges to target.
    _bidi_socket(topology, target_node, reachable_drafter, ip=20)
    _bidi_rdma(topology, target_node, reachable_drafter, iface=50)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000,
        eligible_nodes=[unreachable_drafter, reachable_drafter],
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            unreachable_drafter: create_node_memory(32_000_000_000),
            reachable_drafter: create_node_memory(32_000_000_000),
        },
        {
            target_node: create_node_network(),
            unreachable_drafter: create_node_network(),
            reachable_drafter: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    instance = next(iter(placements.values()))
    assert instance.drafter_placement is not None
    assert instance.drafter_placement.drafter_node_id == reachable_drafter
    assert not degradations  # successful placement, no degradation


def test_asymmetric_skips_drafter_node_without_memory_entry() -> None:
    """Reachable drafter node hasn't reported memory yet => degrade gracefully.

    A freshly-online node can be in the topology with valid edges but
    not yet have a ``MemoryUsage`` entry in ``node_memory`` (the worker
    just hasn't reported its first liveness payload). Previously this
    raised ``KeyError`` deep inside the degradation-detail string,
    aborting placement instead of emitting the
    ``DrafterPlacementDegraded`` event the placement contract promises
    in this branch. Now the detail string explains the missing-stats
    skip explicitly so the operator can wait or pick a different
    eligible node.
    """
    target_node = NodeId()
    drafter_node = NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(drafter_node)
    _bidi_socket(topology, target_node, drafter_node, ip=70)
    _bidi_rdma(topology, target_node, drafter_node, iface=80)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        # Drafter node is intentionally absent from node_memory.
        {target_node: create_node_memory(64_000_000_000)},
        {
            target_node: create_node_network(),
            drafter_node: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    instance = next(iter(placements.values()))
    assert instance.drafter_placement is None
    assert len(degradations) == 1
    assert (
        degradations[0].reason
        == DrafterPlacementDegradationReason.InsufficientDrafterMemory
    )
    assert "has not reported memory stats yet" in degradations[0].detail


def test_asymmetric_continues_scanning_after_first_candidate_below_floor() -> None:
    """First reachable drafter is below memory floor, second is viable
    => placement uses the second.

    Previously the selector pinned ``drafter_node_id = reachable[0]``
    and gave up on the entire reachable list as soon as the first
    candidate failed the memory check. In a cluster where the first
    eligible/reachable node is memory-constrained but later candidates
    are viable, this silently disabled asymmetric drafting. The
    selector now scans all reachable candidates in order and picks the
    first one that meets the memory floor.
    """
    target_node = NodeId()
    constrained_drafter = NodeId()
    viable_drafter = NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(constrained_drafter)
    topology.add_node(viable_drafter)
    # Both candidates are reachable (socket + RDMA), so the only
    # discriminator is memory availability.
    _bidi_socket(topology, target_node, constrained_drafter, ip=90)
    _bidi_rdma(topology, target_node, constrained_drafter, iface=100)
    _bidi_socket(topology, target_node, viable_drafter, ip=110)
    _bidi_rdma(topology, target_node, viable_drafter, iface=120)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000,
        eligible_nodes=[constrained_drafter, viable_drafter],
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            constrained_drafter: create_node_memory(2_000_000_000),  # below floor
            viable_drafter: create_node_memory(32_000_000_000),
        },
        {
            target_node: create_node_network(),
            constrained_drafter: create_node_network(),
            viable_drafter: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    instance = next(iter(placements.values()))
    assert instance.drafter_placement is not None
    assert instance.drafter_placement.drafter_node_id == viable_drafter
    assert not degradations  # successful placement, no degradation


def test_asymmetric_continues_scanning_after_first_candidate_missing_memory() -> None:
    """First reachable drafter has no memory entry, second is viable
    => placement uses the second AND no KeyError.

    Combined fix for both flagged issues: previously the selector
    bailed on ``reachable[0]`` AND then dereferenced
    ``node_memory[reachable[0]]`` in the degradation detail, which
    raised ``KeyError`` rather than emitting the degradation event.
    The scanning loop should reach the viable candidate without ever
    indexing the missing entry.
    """
    target_node = NodeId()
    unreported_drafter = NodeId()
    viable_drafter = NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(unreported_drafter)
    topology.add_node(viable_drafter)
    _bidi_socket(topology, target_node, unreported_drafter, ip=130)
    _bidi_rdma(topology, target_node, unreported_drafter, iface=140)
    _bidi_socket(topology, target_node, viable_drafter, ip=150)
    _bidi_rdma(topology, target_node, viable_drafter, iface=160)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000,
        eligible_nodes=[unreported_drafter, viable_drafter],
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    degradations: list[DrafterPlacementDegraded] = []

    placements = place_instance(
        command,
        topology,
        {},
        # ``unreported_drafter`` is intentionally absent from node_memory.
        {
            target_node: create_node_memory(64_000_000_000),
            viable_drafter: create_node_memory(32_000_000_000),
        },
        {
            target_node: create_node_network(),
            unreported_drafter: create_node_network(),
            viable_drafter: create_node_network(),
        },
        on_drafter_placement_degraded=degradations.append,
    )

    instance = next(iter(placements.values()))
    assert instance.drafter_placement is not None
    assert instance.drafter_placement.drafter_node_id == viable_drafter
    assert not degradations  # successful placement, no degradation


def test_asymmetric_round_trip_serialization() -> None:
    """An asymmetric instance round-trips through pydantic serialisation.

    Codex P1.4 (PR #20): single-node targets stay on ``MlxRing`` even
    when an asymmetric drafter is reachable, because the V3+ wire
    runs the drafter over a TCP socket independent of
    ``mx.distributed`` -- ring's lack of ``Group.split`` is irrelevant
    for a single-rank target. The round-trip is therefore exercised
    on ``MlxRingInstance`` here.
    """
    target_node, drafter_node = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(drafter_node)
    _bidi_socket(topology, target_node, drafter_node, ip=30)
    _bidi_rdma(topology, target_node, drafter_node, iface=60)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_node: create_node_network(),
            drafter_node: create_node_network(),
        },
    )
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxRingInstance)
    assert instance.drafter_placement is not None

    dumped = instance.model_dump()
    rehydrated = MlxRingInstance.model_validate(dumped)
    assert rehydrated == instance
    assert rehydrated.drafter_placement is not None
    assert (
        rehydrated.drafter_placement.drafter_node_id
        == instance.drafter_placement.drafter_node_id
    )


class TestAvailableDrafterModelSelection:
    """Codex P1 (PR #20 round-(N+3), placement.py:617): drafter
    auto-download is explicitly skipped during planning, and
    ``DrafterRunner._handle_load`` raises if the chosen weights are
    missing. So when a card lists ``[fast, fallback]`` and only
    ``fallback`` is on disk on the selected drafter node, picking
    ``drafter_candidates[0]`` unconditionally fails startup. Placement
    must prefer an on-disk candidate; if none are available it
    falls back to the first candidate so the failure mode is no
    worse than the pre-fix behaviour.
    """

    def test_prefers_completed_drafter_over_first_candidate(self) -> None:
        from exo.shared.types.worker.downloads import DownloadCompleted
        from exo.shared.types.worker.shards import PipelineShardMetadata

        target_node, drafter_node = NodeId(), NodeId()
        topology = Topology()
        topology.add_node(target_node)
        topology.add_node(drafter_node)
        _bidi_socket(topology, target_node, drafter_node, ip=200)
        _bidi_rdma(topology, target_node, drafter_node, iface=210)

        card = _drafter_aware_card(
            storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
        )
        # Card lists [fast, fallback]; only the *fallback* is on disk.
        fast_id = ModelId("mlx-community/gemma-4-e2b-it-8bit")
        fallback_id = ModelId("mlx-community/gemma-4-e4b-it-8bit")
        assert list(card.drafter_model_ids) == [fast_id, fallback_id]

        fallback_card = ModelCard(
            model_id=fallback_id,
            storage_size=Memory.from_mb(50),
            n_layers=12,
            hidden_size=768,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        )
        fallback_shard = PipelineShardMetadata(
            model_card=fallback_card,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=fallback_card.n_layers,
            n_layers=fallback_card.n_layers,
        )
        download_status = {
            drafter_node: [
                DownloadCompleted(
                    shard_metadata=fallback_shard,
                    node_id=drafter_node,
                    total=Memory.from_mb(50),
                    model_directory=f"/fake/{fallback_id}",
                ),
            ],
        }

        command = PlaceInstance(
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            command_id=CommandId(),
            model_card=card,
            min_nodes=1,
        )
        degradations: list[DrafterPlacementDegraded] = []

        placements = place_instance(
            command,
            topology,
            {},
            {
                target_node: create_node_memory(64_000_000_000),
                drafter_node: create_node_memory(32_000_000_000),
            },
            {
                target_node: create_node_network(),
                drafter_node: create_node_network(),
            },
            on_drafter_placement_degraded=degradations.append,
            download_status=download_status,
        )

        instance = next(iter(placements.values()))
        assert instance.drafter_placement is not None
        assert instance.drafter_placement.drafter_model_id == fallback_id, (
            f"placement must pick the on-disk fallback drafter; got "
            f"{instance.drafter_placement.drafter_model_id!r}"
        )
        assert not degradations

    def test_prefers_warm_drafter_node_over_cold_node(self) -> None:
        """Codex P1 (PR #20 round-(N+10), placement.py:599):
        memory-eligible nodes are equal candidates only on memory.
        When two memory-eligible nodes are reachable but only one
        has any drafter candidate on disk, placement MUST pick the
        warm (on-disk) node first. Pre-fix it stopped at the first
        memory-eligible reachable candidate (graph order), which
        could be the cold node, and ``DrafterRunner._handle_load``
        then failed startup with ``FileNotFoundError`` because
        drafter auto-download is explicitly skipped during
        planning. After the fix, a warm node always wins over a
        cold one when both are otherwise eligible.
        """
        from exo.shared.types.worker.downloads import DownloadCompleted
        from exo.shared.types.worker.shards import PipelineShardMetadata

        target_node = NodeId()
        cold_drafter_node = NodeId()
        warm_drafter_node = NodeId()
        topology = Topology()
        topology.add_node(target_node)
        topology.add_node(cold_drafter_node)
        topology.add_node(warm_drafter_node)
        # Both candidates fully reachable.
        _bidi_socket(topology, target_node, cold_drafter_node, ip=240)
        _bidi_rdma(topology, target_node, cold_drafter_node, iface=241)
        _bidi_socket(topology, target_node, warm_drafter_node, ip=242)
        _bidi_rdma(topology, target_node, warm_drafter_node, iface=243)

        card = _drafter_aware_card(
            storage_bytes=20_000_000_000,
            eligible_nodes=[cold_drafter_node, warm_drafter_node],
        )
        fast_id = ModelId("mlx-community/gemma-4-e2b-it-8bit")
        # Only the warm node has any drafter weights on disk.
        fast_card = ModelCard(
            model_id=fast_id,
            storage_size=Memory.from_mb(50),
            n_layers=12,
            hidden_size=768,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        )
        fast_shard = PipelineShardMetadata(
            model_card=fast_card,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=fast_card.n_layers,
            n_layers=fast_card.n_layers,
        )
        download_status = {
            warm_drafter_node: [
                DownloadCompleted(
                    shard_metadata=fast_shard,
                    node_id=warm_drafter_node,
                    total=Memory.from_mb(50),
                    model_directory=f"/fake/{fast_id}",
                ),
            ],
            cold_drafter_node: [],
        }

        command = PlaceInstance(
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            command_id=CommandId(),
            model_card=card,
            min_nodes=1,
        )
        degradations: list[DrafterPlacementDegraded] = []

        placements = place_instance(
            command,
            topology,
            {},
            {
                target_node: create_node_memory(64_000_000_000),
                cold_drafter_node: create_node_memory(32_000_000_000),
                warm_drafter_node: create_node_memory(32_000_000_000),
            },
            {
                target_node: create_node_network(),
                cold_drafter_node: create_node_network(),
                warm_drafter_node: create_node_network(),
            },
            on_drafter_placement_degraded=degradations.append,
            download_status=download_status,
        )

        instance = next(iter(placements.values()))
        assert instance.drafter_placement is not None
        assert instance.drafter_placement.drafter_node_id == warm_drafter_node, (
            "placement must prefer the warm drafter node (one with "
            "drafter weights on disk) over an equivalent cold node so "
            "DrafterRunner._handle_load doesn't raise "
            "FileNotFoundError when auto-download is skipped during "
            "planning; got "
            f"{instance.drafter_placement.drafter_node_id!r}"
        )
        assert not degradations

    def test_falls_back_to_first_candidate_when_none_on_disk(self) -> None:
        # No drafter weights on disk anywhere -> placement still picks
        # the first candidate so the runner can surface a load error
        # (the failure mode is unchanged from pre-fix).
        target_node, drafter_node = NodeId(), NodeId()
        topology = Topology()
        topology.add_node(target_node)
        topology.add_node(drafter_node)
        _bidi_socket(topology, target_node, drafter_node, ip=220)
        _bidi_rdma(topology, target_node, drafter_node, iface=230)

        card = _drafter_aware_card(
            storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
        )
        fast_id = ModelId("mlx-community/gemma-4-e2b-it-8bit")

        command = PlaceInstance(
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            command_id=CommandId(),
            model_card=card,
            min_nodes=1,
        )

        placements = place_instance(
            command,
            topology,
            {},
            {
                target_node: create_node_memory(64_000_000_000),
                drafter_node: create_node_memory(32_000_000_000),
            },
            {
                target_node: create_node_network(),
                drafter_node: create_node_network(),
            },
            download_status={},
        )

        instance = next(iter(placements.values()))
        assert instance.drafter_placement is not None
        assert instance.drafter_placement.drafter_model_id == fast_id


class TestDrafterReachabilityDirectional:
    """Codex P1 (PR #20 round-(N+7), placement.py): the v3+ wire is
    unidirectional -- the drafter ALWAYS dials target rank 0 (target
    rank 0 listens, drafter connects). The reachability check must
    validate exactly that direction; pre-fix the round-(N+3) relaxation
    accepted "either direction", which admitted unreachable hosts in
    topologies that recorded only ``target -> drafter`` edges.
    Bootstrap then failed during the actual ``connect()`` instead of
    emitting the intended graceful ``DrafterPlacementDegraded``
    fallback.

    These tests cover the three edge configurations:
    1. Drafter -> target rank 0 only: reachable (matches runtime dial).
    2. Target rank 0 -> drafter only: NOT reachable (wrong direction).
    3. No socket edge in either direction: NOT reachable.
    """

    def test_reachable_with_drafter_to_target_socket_edge(self) -> None:
        # The runtime wire dials drafter -> target rank 0; this
        # direction must satisfy the placement check.
        target_node, drafter_node = NodeId(), NodeId()
        topology = Topology()
        topology.add_node(target_node)
        topology.add_node(drafter_node)
        topology.add_connection(
            Connection(
                source=drafter_node,
                sink=target_node,
                edge=create_socket_connection(300),
            )
        )

        card = _drafter_aware_card(
            storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
        )
        command = PlaceInstance(
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            command_id=CommandId(),
            model_card=card,
            min_nodes=1,
        )
        degradations: list[DrafterPlacementDegraded] = []

        placements = place_instance(
            command,
            topology,
            {},
            {
                target_node: create_node_memory(64_000_000_000),
                drafter_node: create_node_memory(32_000_000_000),
            },
            {
                target_node: create_node_network(),
                drafter_node: create_node_network(),
            },
            on_drafter_placement_degraded=degradations.append,
        )

        instance = next(iter(placements.values()))
        assert instance.drafter_placement is not None, (
            "drafter -> target rank 0 directed edge must satisfy "
            "v3 wire reachability (matches runtime dial direction); "
            f"got degradations={[d.reason.value for d in degradations]!r}"
        )
        assert instance.drafter_placement.drafter_node_id == drafter_node

    def test_not_reachable_with_only_target_to_drafter_socket_edge(self) -> None:
        # Codex P1 (PR #20 round-(N+7), placement.py): a topology that
        # only records the target -> drafter direction does NOT prove
        # the drafter can dial target rank 0. The runtime dial would
        # fail during ``connect()``; placement must surface that as a
        # graceful degradation rather than admitting the host.
        # ``Topology.get_all_connections_between(source, sink)`` is
        # itself directional, so the reverse-only case is genuinely
        # unreachable from the wire's perspective.
        target_node, drafter_node = NodeId(), NodeId()
        topology = Topology()
        topology.add_node(target_node)
        topology.add_node(drafter_node)
        topology.add_connection(
            Connection(
                source=target_node,
                sink=drafter_node,
                edge=create_socket_connection(310),
            )
        )

        card = _drafter_aware_card(
            storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
        )
        command = PlaceInstance(
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            command_id=CommandId(),
            model_card=card,
            min_nodes=1,
        )
        degradations: list[DrafterPlacementDegraded] = []

        place_instance(
            command,
            topology,
            {},
            {
                target_node: create_node_memory(64_000_000_000),
                drafter_node: create_node_memory(32_000_000_000),
            },
            {
                target_node: create_node_network(),
                drafter_node: create_node_network(),
            },
            on_drafter_placement_degraded=degradations.append,
        )

        # No drafter-to-target socket edge -> graceful degradation
        # MUST fire. Without the directional fix, placement would
        # admit this drafter and bootstrap would fail later during
        # the actual ``connect()`` call.
        assert any(
            d.reason
            == DrafterPlacementDegradationReason.NoReachablePathFromTargetRankZero
            for d in degradations
        ), (
            "topology with only target -> drafter edge must emit "
            "NoReachablePathFromTargetRankZero degradation (the runtime "
            "wire dials drafter -> target rank 0, which does not exist "
            f"in this topology); got reasons="
            f"{[d.reason.value for d in degradations]!r}"
        )

    def test_unreachable_when_no_socket_edge_in_either_direction(self) -> None:
        # Defensive: when there's NO socket edge in either direction,
        # placement must still degrade gracefully -- the relaxation
        # only removes the both-directions requirement, not the
        # any-direction requirement.
        target_node, drafter_node = NodeId(), NodeId()
        topology = Topology()
        topology.add_node(target_node)
        topology.add_node(drafter_node)
        # Only RDMA, no socket -- v3 wire is socket-only.
        _bidi_rdma(topology, target_node, drafter_node, iface=320)

        card = _drafter_aware_card(
            storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
        )
        command = PlaceInstance(
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            command_id=CommandId(),
            model_card=card,
            min_nodes=1,
        )
        degradations: list[DrafterPlacementDegraded] = []

        place_instance(
            command,
            topology,
            {},
            {
                target_node: create_node_memory(64_000_000_000),
                drafter_node: create_node_memory(32_000_000_000),
            },
            {
                target_node: create_node_network(),
                drafter_node: create_node_network(),
            },
            on_drafter_placement_degraded=degradations.append,
        )

        # The placement may still succeed without an asymmetric
        # drafter (single-node fallback), but a degradation event
        # MUST surface the no-socket-path.
        assert any(
            d.reason
            == DrafterPlacementDegradationReason.NoReachablePathFromTargetRankZero
            for d in degradations
        ), (
            "no socket edge in either direction must produce "
            "NoReachablePathFromTargetRankZero degradation"
        )


def test_asymmetric_all_node_to_runner_includes_drafter_for_disconnect_check() -> None:
    """``all_node_to_runner`` must list the drafter node so the master's
    instance-deletion loop tears the placement down when the drafter node
    leaves the topology.

    This pins the contract that the master's ``connected_node_ids``
    check at ``master/main.py`` relies on. Iterating
    ``shard_assignments.node_to_runner`` (target ranks only) would
    leave the surviving target runners blocked indefinitely on
    ``transport.forward`` against a dead socket when the drafter node
    disconnects -- the dead-wire ``RemoteTransport.is_failed`` flag
    is set on root only, and non-root has no out-of-band signal that
    the spec loop should abort. Tearing the instance down on drafter-
    node disconnect is the only consistent recovery path.
    """
    target_node, drafter_node = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(target_node)
    topology.add_node(drafter_node)
    _bidi_socket(topology, target_node, drafter_node, ip=2)
    _bidi_rdma(topology, target_node, drafter_node, iface=4)

    card = _drafter_aware_card(
        storage_bytes=20_000_000_000, eligible_nodes=[drafter_node]
    )
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )
    placements = place_instance(
        command,
        topology,
        {},
        {
            target_node: create_node_memory(64_000_000_000),
            drafter_node: create_node_memory(32_000_000_000),
        },
        {
            target_node: create_node_network(),
            drafter_node: create_node_network(),
        },
    )
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert instance.drafter_placement is not None

    # Both nodes must appear in ``all_node_to_runner`` so the master's
    # disconnect check fires for either one.
    assert target_node in instance.all_node_to_runner
    assert drafter_node in instance.all_node_to_runner
    assert (
        instance.all_node_to_runner[drafter_node]
        == instance.drafter_placement.drafter_runner_id
    )

    # The legacy mapping (target shards only) intentionally excludes
    # the drafter; this is the bug the master fix addresses by
    # iterating ``all_node_to_runner`` instead.
    assert target_node in instance.shard_assignments.node_to_runner
    assert drafter_node not in instance.shard_assignments.node_to_runner


def test_asymmetric_drafter_and_target_peer_ports_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``drafter_socket_port`` and ``target_peer_socket_port`` must
    never be allocated to the same port.

    Both ports are drawn from the same ~13K-wide ephemeral range
    (49153-65535 minus the master API port 52415), so two independent
    random draws can occasionally collide -- on collision, one of the
    two listener binds fails with EADDRINUSE during runner bootstrap
    (drafter accept loop in ``_maybe_accept_drafter_socket`` versus
    target peer fanout in ``_maybe_setup_target_peer_fanout``),
    causing a nondeterministic instance failure under asymmetric
    multi-target placements.

    Test deterministically forces a collision: ``random.randint`` is
    monkeypatched to return the same port the first two times it's
    called, then a different port on the third call. The placement
    code must observe the collision and re-roll, producing two
    distinct ports.
    """
    # Placement allocates ports in this order:
    #   1. ``pre_allocated_listener_port`` (jaccl coordinator port) --
    #      first ``random_ephemeral_port`` call.
    #   2. ``drafter_socket_port`` -- via ``random_ephemeral_port_excluding``
    #      (which calls ``random_ephemeral_port`` until it finds a
    #      port outside ``reserved_ports``).
    #   3. ``target_peer_socket_port`` -- ditto, also avoiding
    #      ``drafter_socket_port``.
    #
    # Force a collision between drafter and target peer: drafter and
    # target peer both draw 60001 first; target peer's exclusion loop
    # re-rolls to 60002.
    sequence: list[int] = [
        59000,  # pre_allocated_listener_port (jaccl coordinator)
        60001,  # drafter_socket_port
        60001,  # target_peer_socket_port -- COLLISION with drafter
        60002,  # target_peer re-roll, distinct
    ]
    # Pad with distinct values for any further calls.
    sequence.extend(50000 + i for i in range(1, 20))
    drawn_ports = iter(sequence)

    def fake_random_ephemeral_port() -> int:
        return next(drawn_ports)

    # Patch at the source module so ``random_ephemeral_port_excluding``
    # (which lives in ``exo.utils.ports`` and calls its own local
    # ``random_ephemeral_port``) also sees the patched sequence.
    monkeypatch.setattr(
        "exo.utils.ports.random_ephemeral_port",
        fake_random_ephemeral_port,
    )
    monkeypatch.setattr(
        "exo.master.placement.random_ephemeral_port",
        fake_random_ephemeral_port,
    )

    target_a, target_b, drafter_node = NodeId(), NodeId(), NodeId()
    topology = Topology()
    for n in (target_a, target_b, drafter_node):
        topology.add_node(n)
    _bidi_rdma(topology, target_a, target_b, iface=10)
    _bidi_socket(topology, target_a, target_b, ip=12)
    _bidi_rdma(topology, target_a, drafter_node, iface=20)
    _bidi_rdma(topology, target_b, drafter_node, iface=22)
    _bidi_socket(topology, target_a, drafter_node, ip=14)
    _bidi_socket(topology, target_b, drafter_node, ip=16)

    card = _drafter_aware_card(
        storage_bytes=40_000_000_000,
        eligible_nodes=[drafter_node],
        family="qwen",
        base_model="Qwen3 30B",
        model_id="mlx-community/Qwen3-30B-A3B-4bit",
    )
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=card,
        min_nodes=2,
    )
    memory = {
        target_a: create_node_memory(32_000_000_000),
        target_b: create_node_memory(32_000_000_000),
        drafter_node: create_node_memory(32_000_000_000),
    }
    network = {
        target_a: create_node_network(),
        target_b: create_node_network(),
        drafter_node: create_node_network(),
    }

    placements = place_instance(
        command,
        topology,
        {},
        memory,
        network,
        node_rdma_ctl={
            target_a: NodeRdmaCtlStatus(enabled=True),
            target_b: NodeRdmaCtlStatus(enabled=True),
            drafter_node: NodeRdmaCtlStatus(enabled=True),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert instance.drafter_placement is not None
    placement = instance.drafter_placement

    # The collision was observed and re-rolled.
    assert placement.drafter_socket_port == 60001
    assert placement.target_peer_socket_port == 60002
    assert placement.drafter_socket_port != placement.target_peer_socket_port


def test_drafter_and_target_peer_avoid_jaccl_coordinator_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex P2 (PR #21 round 3): the original collision-avoidance loop
    only checked ``target_peer_socket_port != drafter_socket_port``,
    so a draw that happened to coincide with the jaccl coordinator
    port (or the ring ephemeral port) would slip through and fail at
    bind with ``EADDRINUSE`` during runner bootstrap. The fix
    pre-allocates the per-meta listener port and threads it as a
    ``reserved_ports`` set into ``_select_drafter_placement`` so all
    rank-0 listener ports are drawn distinct.

    Test deterministically forces ``drafter_socket_port`` to collide
    with the pre-allocated jaccl coordinator port. The fix must
    re-roll until distinct.
    """
    # Allocation order (with the fix in place):
    #   1. pre_allocated_listener_port -> 60100 (becomes the jaccl
    #      coordinator port)
    #   2. drafter_socket_port via random_ephemeral_port_excluding
    #      (reserved={60100}) -- first draw 60100 collides, re-roll
    #      to 60101
    #   3. target_peer_socket_port via random_ephemeral_port_excluding
    #      (reserved={60100, 60101}) -- first draw 60101 collides,
    #      re-roll to 60100 collides, re-roll to 60102
    sequence: list[int] = [
        60100,  # pre_allocated_listener_port (becomes jaccl coordinator)
        60100,  # drafter draw 1 -- collides with reserved {60100}
        60101,  # drafter draw 2 -- accepted
        60101,  # target_peer draw 1 -- collides with drafter
        60100,  # target_peer draw 2 -- collides with reserved
        60102,  # target_peer draw 3 -- accepted
    ]
    sequence.extend(50000 + i for i in range(1, 20))
    drawn_ports = iter(sequence)

    def fake_random_ephemeral_port() -> int:
        return next(drawn_ports)

    monkeypatch.setattr(
        "exo.utils.ports.random_ephemeral_port",
        fake_random_ephemeral_port,
    )
    monkeypatch.setattr(
        "exo.master.placement.random_ephemeral_port",
        fake_random_ephemeral_port,
    )

    target_a, target_b, drafter_node = NodeId(), NodeId(), NodeId()
    topology = Topology()
    for n in (target_a, target_b, drafter_node):
        topology.add_node(n)
    _bidi_rdma(topology, target_a, target_b, iface=10)
    _bidi_socket(topology, target_a, target_b, ip=12)
    _bidi_rdma(topology, target_a, drafter_node, iface=20)
    _bidi_rdma(topology, target_b, drafter_node, iface=22)
    _bidi_socket(topology, target_a, drafter_node, ip=14)
    _bidi_socket(topology, target_b, drafter_node, ip=16)

    card = _drafter_aware_card(
        storage_bytes=40_000_000_000,
        eligible_nodes=[drafter_node],
        family="qwen",
        base_model="Qwen3 30B",
        model_id="mlx-community/Qwen3-30B-A3B-4bit",
    )
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=card,
        min_nodes=2,
    )
    memory = {
        target_a: create_node_memory(32_000_000_000),
        target_b: create_node_memory(32_000_000_000),
        drafter_node: create_node_memory(32_000_000_000),
    }
    network = {
        target_a: create_node_network(),
        target_b: create_node_network(),
        drafter_node: create_node_network(),
    }

    placements = place_instance(
        command,
        topology,
        {},
        memory,
        network,
        node_rdma_ctl={
            target_a: NodeRdmaCtlStatus(enabled=True),
            target_b: NodeRdmaCtlStatus(enabled=True),
            drafter_node: NodeRdmaCtlStatus(enabled=True),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxJacclInstance)
    assert instance.drafter_placement is not None
    placement = instance.drafter_placement

    # All three rank-0 listener ports are mutually distinct.
    # ``jaccl_coordinators`` values are ``"host:port"`` strings.
    coordinator_ports = {
        int(addr.rsplit(":", 1)[1]) for addr in instance.jaccl_coordinators.values()
    }
    assert coordinator_ports == {60100}
    assert placement.drafter_socket_port == 60101
    assert placement.target_peer_socket_port == 60102
    listener_ports = (
        coordinator_ports
        | {placement.drafter_socket_port}
        | {placement.target_peer_socket_port}
    )
    assert len(listener_ports) == 3, (
        "all rank-0 listener ports (jaccl coordinator, drafter accept, "
        "target-peer fanout) must be mutually distinct to avoid "
        "EADDRINUSE during runner bootstrap; got "
        f"{listener_ports!r}"
    )
