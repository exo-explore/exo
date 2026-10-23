from exo.master.placement_utils import (
    _allocate_and_validate_layers,
    filter_cycles_by_memory,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryPressureLevel, MemoryUsage
from exo.shared.types.topology import Cycle


def _make_node_memory(
    available_gb: float,
    pressure: MemoryPressureLevel = MemoryPressureLevel.NORMAL,
) -> MemoryUsage:
    bytes_val = int(available_gb * 1024 * 1024 * 1024)
    return MemoryUsage.from_bytes(
        ram_total=64 * 1024 * 1024 * 1024,
        ram_available=bytes_val,
        swap_total=0,
        swap_available=0,
        pressure_level=pressure,
    )


def test_filter_cycles_rejects_critical_pressure():
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    cycle = Cycle(node_ids=[node_a, node_b])

    # Both nodes normal -> Cycle accepted
    node_mem_normal = {
        node_a: _make_node_memory(20.0, MemoryPressureLevel.NORMAL),
        node_b: _make_node_memory(20.0, MemoryPressureLevel.NORMAL),
    }
    filtered = filter_cycles_by_memory([cycle], node_mem_normal, Memory.from_gb(30))
    assert len(filtered) == 1

    # One node under critical pressure -> Cycle rejected
    node_mem_critical = {
        node_a: _make_node_memory(20.0, MemoryPressureLevel.NORMAL),
        node_b: _make_node_memory(20.0, MemoryPressureLevel.CRITICAL),
    }
    filtered_crit = filter_cycles_by_memory(
        [cycle], node_mem_critical, Memory.from_gb(30)
    )
    assert len(filtered_crit) == 0


def test_allocate_layers_derates_warning_nodes():
    node_a = NodeId("node-a")  # Normal
    node_b = NodeId("node-b")  # Warning
    node_ids = [node_a, node_b]

    # Equal available memory, but node_b is under WARNING pressure
    node_memory = {
        node_a: _make_node_memory(30.0, MemoryPressureLevel.NORMAL),
        node_b: _make_node_memory(30.0, MemoryPressureLevel.WARNING),
    }

    dummy_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_gb(10),
        n_layers=32,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )

    allocations = _allocate_and_validate_layers(
        node_ids=node_ids,
        node_memory=node_memory,
        total_memory=Memory.from_gb(60),
        model_card=dummy_card,
    )

    # Node A (Normal) should get more layers than Node B (Warning)
    # Ratio: 30 / (30*0.6) = 30 / 18 = 1.67x
    assert allocations[0] > allocations[1]
    assert sum(allocations) == 32
