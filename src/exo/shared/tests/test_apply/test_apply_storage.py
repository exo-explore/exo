from exo.shared.apply import apply_node_timed_out, apply_storage_config_updated
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeTimedOut, StorageConfigUpdated
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.storage import StorageConfig

NODE_A = NodeId("node-a")
NODE_B = NodeId("node-b")


def test_storage_config_updated_adds_config() -> None:
    state = State()
    config = StorageConfig(max_storage=Memory.from_gb(10), storage_policy="manual")
    event = StorageConfigUpdated(node_id=NODE_A, storage_config=config)

    new_state = apply_storage_config_updated(event, state)

    assert NODE_A in new_state.node_storage_config
    assert new_state.node_storage_config[NODE_A].max_storage == Memory.from_gb(10)
    assert new_state.node_storage_config[NODE_A].storage_policy == "manual"


def test_storage_config_updated_overwrites_existing() -> None:
    config1 = StorageConfig(max_storage=Memory.from_gb(10), storage_policy="manual")
    state = State(node_storage_config={NODE_A: config1})

    config2 = StorageConfig(max_storage=Memory.from_gb(20), storage_policy="auto-evict")
    event = StorageConfigUpdated(node_id=NODE_A, storage_config=config2)

    new_state = apply_storage_config_updated(event, state)

    assert new_state.node_storage_config[NODE_A].max_storage == Memory.from_gb(20)
    assert new_state.node_storage_config[NODE_A].storage_policy == "auto-evict"


def test_node_timed_out_cleans_up_storage_config() -> None:
    config = StorageConfig(max_storage=Memory.from_gb(10))
    state = State(node_storage_config={NODE_A: config, NODE_B: config})

    event = NodeTimedOut(node_id=NODE_A)
    new_state = apply_node_timed_out(event, state)

    assert NODE_A not in new_state.node_storage_config
    assert NODE_B in new_state.node_storage_config
