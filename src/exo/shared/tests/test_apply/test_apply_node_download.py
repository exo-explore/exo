from exo.shared.apply import apply_node_download_progress, apply_node_timed_out
from exo.shared.tests.conftest import get_pipeline_shard_metadata
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeDownloadProgress, NodeTimedOut
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadPending
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


def test_apply_node_timed_out_removes_pending_downloads() -> None:
    retired_node = NodeId("retired-node")
    connected_node = NodeId("connected-node")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    retired_download = DownloadPending(
        node_id=retired_node,
        shard_metadata=shard,
    )
    connected_download = DownloadPending(
        node_id=connected_node,
        shard_metadata=shard,
    )
    state = State(
        downloads={
            retired_node: [retired_download],
            connected_node: [connected_download],
        }
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=retired_node), state)

    assert new_state.downloads == {connected_node: [connected_download]}
