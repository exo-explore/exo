from exo.shared.apply import apply_node_download_progress
from exo.shared.tests.conftest import get_pipeline_shard_metadata
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadPending,
    DownloadProgress,
)
from exo.worker.tests.constants import MODEL_A_ID, MODEL_B_ID


def test_apply_node_download_progress():
    state = State()
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    event = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard1,
        total=Memory(),
    )

    new_state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=event), state
    )

    assert new_state.downloads == {NodeId("node-1"): [event]}


def test_apply_two_node_download_progress():
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_B_ID, device_rank=0, world_size=2)
    event1 = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard1,
        total=Memory(),
    )
    event2 = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard2,
        total=Memory(),
    )
    state = State(downloads={NodeId("node-1"): [event1]})

    new_state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=event2), state
    )

    assert new_state.downloads == {NodeId("node-1"): [event1, event2]}


def _apply(state: State, progress: DownloadProgress) -> State:
    return apply_node_download_progress(
        NodeDownloadProgress(download_progress=progress), state
    )


def test_a_model_not_on_the_node_is_not_listed():
    # Most of the catalogue is on no node: these entries were 96% of a cluster's state
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    state = State()
    after = _apply(
        state, DownloadPending(node_id=NodeId("node-1"), shard_metadata=shard)
    )
    assert after.downloads == {}
    assert after is state


def test_a_model_whose_files_went_away_is_no_longer_listed():
    node = NodeId("node-1")
    shard_a = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    shard_b = get_pipeline_shard_metadata(MODEL_B_ID, device_rank=0, world_size=1)
    completed_a = DownloadCompleted(
        node_id=node, shard_metadata=shard_a, total=Memory()
    )
    completed_b = DownloadCompleted(
        node_id=node, shard_metadata=shard_b, total=Memory()
    )
    state = State(downloads={node: [completed_a, completed_b]})

    state = _apply(state, DownloadPending(node_id=node, shard_metadata=shard_a))
    assert state.downloads == {node: [completed_b]}
    # A node with nothing left isn't listed either
    state = _apply(state, DownloadPending(node_id=node, shard_metadata=shard_b))
    assert state.downloads == {}


def test_a_partial_download_is_listed():
    node = NodeId("node-1")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    paused = DownloadPending(
        node_id=node,
        shard_metadata=shard,
        downloaded=Memory.from_bytes(1),
        total=Memory.from_bytes(10),
    )
    assert _apply(State(), paused).downloads == {node: [paused]}
