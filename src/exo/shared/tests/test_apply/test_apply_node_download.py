from exo.shared.apply import apply_node_download_progress
from exo.shared.tests.conftest import (
    get_pipeline_shard_metadata,
    get_tensor_shard_metadata,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeDownloadProgress
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


def test_apply_download_progress_replaces_when_shard_metadata_type_changes():
    """When the same model is re-created with a different shard type, the download
    entry should be replaced rather than appended."""
    pipeline_shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    tensor_shard = get_tensor_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)

    old_entry = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=pipeline_shard,
        total=Memory(),
    )
    state = State(downloads={NodeId("node-1"): [old_entry]})

    new_entry = DownloadPending(
        node_id=NodeId("node-1"),
        shard_metadata=tensor_shard,
    )
    new_state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=new_entry), state
    )

    # Should replace, not append - one entry per model per node
    assert len(new_state.downloads[NodeId("node-1")]) == 1
    assert new_state.downloads[NodeId("node-1")][0] == new_entry
