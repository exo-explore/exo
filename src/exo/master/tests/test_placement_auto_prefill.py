"""Tests for auto-prefill placement (multi-GPU prefill spread).

When ``ModelCard.prefill_eligible_nodes`` is non-empty, placement
auto-creates a single-rank prefill-only sibling instance on each viable
node and the master emits an ``InstanceLinkCreated`` linking them to
the decode instance. The link tells ``_prefill_endpoint_for`` to
spread incoming requests' prefill traffic across the linked nodes,
so slot N's TTFT is decoupled from slot 0's prefill (different GPUs,
not different time slots on the same one).

Coverage:
- Sibling placed on a viable eligible node distinct from the decode
  cycle (and distinct from the asymmetric drafter rank when present).
- Drafter and prefill overlap is excluded automatically (chosen drafter
  node is removed from prefill candidates).
- Eligible node not alive in topology -> skipped, no exception.
- Eligible node has insufficient RAM -> skipped, decode still placed,
  no link emitted.
- Empty ``prefill_eligible_nodes`` -> legacy single-instance behaviour
  (backwards compat).
- Recursive sanitisation: the sibling card has no drafter / no further
  prefill spawn (so we don't recurse forever).
"""

from collections.abc import Iterator

import pytest
from loguru import logger as loguru_logger

from exo.master.placement import auto_place_prefill_siblings, place_instance
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
from exo.shared.types.memory import Memory
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding


@pytest.fixture
def loguru_capture() -> Iterator[list[str]]:
    captured: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: captured.append(str(message)), level="WARNING"
    )
    try:
        yield captured
    finally:
        loguru_logger.remove(sink_id)


def _prefill_aware_card(
    *,
    storage_bytes: int,
    prefill_eligible: list[NodeId],
    drafter_eligible: list[NodeId] | None = None,
    drafter_models: list[ModelId] | None = None,
) -> ModelCard:
    return ModelCard(
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        storage_size=Memory.from_bytes(storage_bytes),
        n_layers=60,
        hidden_size=5376,
        num_key_value_heads=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="gemma",
        base_model="Gemma 4 26B",
        drafter_model_ids=drafter_models or [],
        drafter_eligible_nodes=drafter_eligible or [],
        prefill_eligible_nodes=prefill_eligible,
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
        Connection(source=b, sink=a, edge=create_rdma_connection(iface))
    )


def test_prefill_sibling_placed_on_eligible_idle_node() -> None:
    """Decode on smbp + prefill sibling on bmbp -> 2 instances, 1 link.

    The decode instance is single-rank (PP=1) on smbp; bmbp is
    declared as a prefill-eligible idle node. Auto-prefill places a
    single-rank prefill-only sibling on bmbp and the master will
    emit ``InstanceLinkCreated`` linking them.
    """
    smbp = NodeId("smbp")
    bmbp = NodeId("bmbp")
    topology = Topology()
    topology.add_node(smbp)
    topology.add_node(bmbp)
    _bidi_socket(topology, smbp, bmbp, ip=10)
    _bidi_rdma(topology, smbp, bmbp, iface=1)

    node_memory = {
        smbp: create_node_memory(Memory.from_gb(120).in_bytes),
        bmbp: create_node_memory(Memory.from_gb(40).in_bytes),
    }
    node_network = {
        smbp: create_node_network(),
        bmbp: create_node_network(),
    }
    card = _prefill_aware_card(
        storage_bytes=Memory.from_gb(13).in_bytes,
        prefill_eligible=[bmbp],
    )

    decode_placement = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        topology,
        {},
        node_memory,
        node_network,
        required_nodes={smbp},
    )
    assert len(decode_placement) == 1
    decode_id, decode_inst = next(iter(decode_placement.items()))

    siblings, sibling_ids = auto_place_prefill_siblings(
        decode_instance_id=decode_id,
        decode_instance=decode_inst,
        model_card=card,
        topology=topology,
        current_instances=decode_placement,
        node_memory=node_memory,
        node_network=node_network,
    )
    assert len(siblings) == 1
    assert len(sibling_ids) == 1
    sibling = siblings[sibling_ids[0]]
    assert bmbp in sibling.shard_assignments.node_to_runner
    assert smbp not in sibling.shard_assignments.node_to_runner


def test_prefill_excludes_chosen_drafter_node() -> None:
    """Asymmetric decode (smbp+smbpt) + drafter on bmbp -> studio left for prefill.

    With drafter_eligible=[bmbp] and prefill_eligible=[bmbp,studio],
    bmbp gets used as the drafter rank and studio is the only viable
    prefill candidate.
    """
    smbp = NodeId("smbp")
    smbpt = NodeId("smbpt")
    bmbp = NodeId("bmbp")
    studio = NodeId("studio")
    topology = Topology()
    for n in (smbp, smbpt, bmbp, studio):
        topology.add_node(n)
    for a, b, ip in [
        (smbp, smbpt, 10),
        (smbp, bmbp, 12),
        (smbp, studio, 14),
        (smbpt, bmbp, 16),
        (smbpt, studio, 18),
        (bmbp, studio, 20),
    ]:
        _bidi_socket(topology, a, b, ip=ip)
    for a, b, iface in [
        (smbp, smbpt, 1),
        (smbp, bmbp, 2),
        (smbpt, bmbp, 3),
    ]:
        _bidi_rdma(topology, a, b, iface=iface)

    node_memory = {
        smbp: create_node_memory(Memory.from_gb(120).in_bytes),
        smbpt: create_node_memory(Memory.from_gb(120).in_bytes),
        bmbp: create_node_memory(Memory.from_gb(40).in_bytes),
        studio: create_node_memory(Memory.from_gb(120).in_bytes),
    }
    node_network = {n: create_node_network() for n in (smbp, smbpt, bmbp, studio)}

    card = _prefill_aware_card(
        storage_bytes=Memory.from_gb(13).in_bytes,
        prefill_eligible=[bmbp, studio],
        drafter_eligible=[bmbp],
        drafter_models=[ModelId("mlx-community/gemma-4-e2b-it-4bit")],
    )

    decode_placement = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        topology,
        {},
        node_memory,
        node_network,
        required_nodes={smbp},
    )
    decode_id, decode_inst = next(iter(decode_placement.items()))
    assert decode_inst.drafter_placement is not None
    assert decode_inst.drafter_placement.drafter_node_id == bmbp

    siblings, sibling_ids = auto_place_prefill_siblings(
        decode_instance_id=decode_id,
        decode_instance=decode_inst,
        model_card=card,
        topology=topology,
        current_instances=decode_placement,
        node_memory=node_memory,
        node_network=node_network,
    )
    assert len(siblings) == 1
    sibling = siblings[sibling_ids[0]]
    sibling_nodes = set(sibling.shard_assignments.node_to_runner.keys())
    assert sibling_nodes == {studio}, (
        f"prefill sibling should land on studio (not the drafter node bmbp); "
        f"got nodes={sibling_nodes}"
    )


def test_prefill_skipped_when_eligible_node_offline(
    loguru_capture: list[str],
) -> None:
    """Eligible node not in topology -> no sibling, no exception."""
    smbp = NodeId("smbp")
    ghost = NodeId("ghost-not-in-topology")
    topology = Topology()
    topology.add_node(smbp)
    node_memory = {smbp: create_node_memory(Memory.from_gb(120).in_bytes)}
    node_network = {smbp: create_node_network()}

    card = _prefill_aware_card(
        storage_bytes=Memory.from_gb(13).in_bytes,
        prefill_eligible=[ghost],
    )
    decode_placement = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        topology,
        {},
        node_memory,
        node_network,
        required_nodes={smbp},
    )
    decode_id, decode_inst = next(iter(decode_placement.items()))

    siblings, sibling_ids = auto_place_prefill_siblings(
        decode_instance_id=decode_id,
        decode_instance=decode_inst,
        model_card=card,
        topology=topology,
        current_instances=decode_placement,
        node_memory=node_memory,
        node_network=node_network,
    )
    assert siblings == {}
    assert sibling_ids == []
    assert any("Auto-prefill placement skipped" in m for m in loguru_capture), (
        loguru_capture
    )


def test_prefill_skipped_when_eligible_node_oom(loguru_capture: list[str]) -> None:
    """Eligible node lacks RAM -> placement raises and is logged-and-skipped."""
    smbp = NodeId("smbp")
    tiny = NodeId("tiny")
    topology = Topology()
    topology.add_node(smbp)
    topology.add_node(tiny)
    _bidi_socket(topology, smbp, tiny, ip=10)
    node_memory = {
        smbp: create_node_memory(Memory.from_gb(120).in_bytes),
        tiny: create_node_memory(Memory.from_gb(2).in_bytes),
    }
    node_network = {smbp: create_node_network(), tiny: create_node_network()}

    card = _prefill_aware_card(
        storage_bytes=Memory.from_gb(13).in_bytes,
        prefill_eligible=[tiny],
    )
    decode_placement = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        topology,
        {},
        node_memory,
        node_network,
        required_nodes={smbp},
    )
    decode_id, decode_inst = next(iter(decode_placement.items()))

    siblings, sibling_ids = auto_place_prefill_siblings(
        decode_instance_id=decode_id,
        decode_instance=decode_inst,
        model_card=card,
        topology=topology,
        current_instances=decode_placement,
        node_memory=node_memory,
        node_network=node_network,
    )
    assert siblings == {}
    assert sibling_ids == []
    assert any("Auto-prefill skip" in m for m in loguru_capture), loguru_capture


def test_empty_prefill_eligible_preserves_legacy_path() -> None:
    """No ``prefill_eligible_nodes`` -> auto-prefill is a no-op."""
    smbp = NodeId("smbp")
    bmbp = NodeId("bmbp")
    topology = Topology()
    topology.add_node(smbp)
    topology.add_node(bmbp)
    _bidi_socket(topology, smbp, bmbp, ip=10)
    node_memory = {
        smbp: create_node_memory(Memory.from_gb(120).in_bytes),
        bmbp: create_node_memory(Memory.from_gb(40).in_bytes),
    }
    node_network = {smbp: create_node_network(), bmbp: create_node_network()}

    card = _prefill_aware_card(
        storage_bytes=Memory.from_gb(13).in_bytes,
        prefill_eligible=[],
    )
    decode_placement = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        topology,
        {},
        node_memory,
        node_network,
        required_nodes={smbp},
    )
    decode_id, decode_inst = next(iter(decode_placement.items()))

    siblings, sibling_ids = auto_place_prefill_siblings(
        decode_instance_id=decode_id,
        decode_instance=decode_inst,
        model_card=card,
        topology=topology,
        current_instances=decode_placement,
        node_memory=node_memory,
        node_network=node_network,
    )
    assert siblings == {}
    assert sibling_ids == []


def test_prefill_sibling_does_not_carry_drafter() -> None:
    """The recursive sub-placement uses a drafter-cleared card.

    Even though the model card declares a drafter, the prefill sibling
    has ``drafter_placement is None`` (it's a TCP prefill server, not
    a decode instance, so it has no use for a drafter).
    """
    smbp = NodeId("smbp")
    bmbp = NodeId("bmbp")
    studio = NodeId("studio")
    topology = Topology()
    for n in (smbp, bmbp, studio):
        topology.add_node(n)
    for a, b, ip in [(smbp, bmbp, 10), (smbp, studio, 12), (bmbp, studio, 14)]:
        _bidi_socket(topology, a, b, ip=ip)

    node_memory = {
        smbp: create_node_memory(Memory.from_gb(120).in_bytes),
        bmbp: create_node_memory(Memory.from_gb(40).in_bytes),
        studio: create_node_memory(Memory.from_gb(120).in_bytes),
    }
    node_network = {n: create_node_network() for n in (smbp, bmbp, studio)}

    card = _prefill_aware_card(
        storage_bytes=Memory.from_gb(13).in_bytes,
        prefill_eligible=[studio],
        drafter_eligible=[bmbp],
        drafter_models=[ModelId("mlx-community/gemma-4-e2b-it-4bit")],
    )
    decode_placement = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        topology,
        {},
        node_memory,
        node_network,
        required_nodes={smbp},
    )
    decode_id, decode_inst = next(iter(decode_placement.items()))

    siblings, sibling_ids = auto_place_prefill_siblings(
        decode_instance_id=decode_id,
        decode_instance=decode_inst,
        model_card=card,
        topology=topology,
        current_instances=decode_placement,
        node_memory=node_memory,
        node_network=node_network,
    )
    assert len(siblings) == 1
    sibling = siblings[sibling_ids[0]]
    assert sibling.drafter_placement is None, (
        "prefill sibling must not own a drafter -- only the decode does"
    )


def test_eligible_duplicates_are_deduped() -> None:
    """``prefill_eligible_nodes=[bmbp, bmbp]`` -> one sibling, not two."""
    smbp = NodeId("smbp")
    bmbp = NodeId("bmbp")
    topology = Topology()
    topology.add_node(smbp)
    topology.add_node(bmbp)
    _bidi_socket(topology, smbp, bmbp, ip=10)
    node_memory = {
        smbp: create_node_memory(Memory.from_gb(120).in_bytes),
        bmbp: create_node_memory(Memory.from_gb(40).in_bytes),
    }
    node_network = {smbp: create_node_network(), bmbp: create_node_network()}

    card = _prefill_aware_card(
        storage_bytes=Memory.from_gb(13).in_bytes,
        prefill_eligible=[bmbp, bmbp],
    )
    decode_placement = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        topology,
        {},
        node_memory,
        node_network,
        required_nodes={smbp},
    )
    decode_id, decode_inst = next(iter(decode_placement.items()))

    siblings, sibling_ids = auto_place_prefill_siblings(
        decode_instance_id=decode_id,
        decode_instance=decode_inst,
        model_card=card,
        topology=topology,
        current_instances=decode_placement,
        node_memory=node_memory,
        node_network=node_network,
    )
    assert len(siblings) == 1
    assert len(sibling_ids) == 1
