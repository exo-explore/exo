from exo.routing import topics
from exo.shared.types.commands import ForwarderCommand, RequestSnapshot
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.snapshots import SnapshotChunk, SnapshotTransferId


def test_request_snapshot_round_trips_through_forwarder_command() -> None:
    command = ForwarderCommand(
        origin=SystemId("system-1"),
        command=RequestSnapshot(requester_node_id=NodeId("worker-1")),
    )

    restored = ForwarderCommand.model_validate_json(command.model_dump_json())

    assert isinstance(restored.command, RequestSnapshot)
    assert restored.command.requester_node_id == NodeId("worker-1")


def test_snapshot_response_topic_round_trips_chunk() -> None:
    chunk = SnapshotChunk.from_data(
        data=b"snapshot-bytes",
        transfer_id=SnapshotTransferId("transfer-1"),
        requester_node_id=NodeId("worker-1"),
        session_id=SessionId(master_node_id=NodeId("master"), election_clock=1),
        schema_version=1,
        last_event_applied_idx=42,
        chunk_index=0,
        total_chunks=1,
        sha256_hex="unused",
    )

    restored = topics.SNAPSHOT_RESPONSES.deserialize(
        topics.SNAPSHOT_RESPONSES.serialize(chunk)
    )

    assert restored == chunk
    assert restored.data == b"snapshot-bytes"
