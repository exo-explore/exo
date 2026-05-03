import hashlib

import pytest
import zstandard

from exo.routing.snapshot_receiver import SnapshotReceiver
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.snapshots import SnapshotChunk, SnapshotTransferId
from exo.shared.types.state import State


@pytest.fixture
def session_id() -> SessionId:
    return SessionId(master_node_id=NodeId("master"), election_clock=0)


@pytest.fixture
def my_node() -> NodeId:
    return NodeId("worker-1")


def _encode(state: State) -> bytes:
    return zstandard.ZstdCompressor().compress(state.model_dump_json().encode("utf-8"))


def _make_chunks(
    body: bytes,
    *,
    chunk_size: int,
    requester_node_id: NodeId,
    session_id: SessionId,
    state: State,
    transfer_id: SnapshotTransferId | None = None,
) -> list[SnapshotChunk]:
    sha256 = hashlib.sha256(body).hexdigest()
    transfer_id = transfer_id or SnapshotTransferId()
    pieces = [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)] or [
        b""
    ]
    return [
        SnapshotChunk.from_data(
            data=piece,
            transfer_id=transfer_id,
            requester_node_id=requester_node_id,
            session_id=session_id,
            schema_version=state.schema_version,
            last_event_applied_idx=state.last_event_applied_idx,
            chunk_index=i,
            total_chunks=len(pieces),
            sha256_hex=sha256,
        )
        for i, piece in enumerate(pieces)
    ]


def test_completes_on_full_transfer(my_node: NodeId, session_id: SessionId):
    state = State(last_event_applied_idx=42)
    chunks = _make_chunks(
        _encode(state),
        chunk_size=64,
        requester_node_id=my_node,
        session_id=session_id,
        state=state,
    )
    receiver = SnapshotReceiver(my_node, session_id)

    received = None
    for chunk in chunks:
        received = receiver.ingest(chunk)
    assert received is not None
    assert received.last_event_applied_idx == 42
    assert received.state.last_event_applied_idx == 42


def test_handles_out_of_order_chunks(my_node: NodeId, session_id: SessionId):
    state = State(last_event_applied_idx=99)
    chunks = _make_chunks(
        _encode(state),
        chunk_size=32,
        requester_node_id=my_node,
        session_id=session_id,
        state=state,
    )
    receiver = SnapshotReceiver(my_node, session_id)

    # Reverse them.
    received = None
    for chunk in reversed(chunks):
        received = receiver.ingest(chunk)
    assert received is not None
    assert received.last_event_applied_idx == 99


def test_ignores_chunks_for_other_recipients(my_node: NodeId, session_id: SessionId):
    state = State(last_event_applied_idx=1)
    other = NodeId("worker-2")
    chunks = _make_chunks(
        _encode(state),
        chunk_size=64,
        requester_node_id=other,
        session_id=session_id,
        state=state,
    )
    receiver = SnapshotReceiver(my_node, session_id)
    for chunk in chunks:
        assert receiver.ingest(chunk) is None


def test_ignores_chunks_from_stale_session(my_node: NodeId, session_id: SessionId):
    state = State(last_event_applied_idx=1)
    other_session = SessionId(master_node_id=NodeId("other-master"), election_clock=99)
    chunks = _make_chunks(
        _encode(state),
        chunk_size=64,
        requester_node_id=my_node,
        session_id=other_session,
        state=state,
    )
    receiver = SnapshotReceiver(my_node, session_id)
    for chunk in chunks:
        assert receiver.ingest(chunk) is None


def test_discards_on_checksum_mismatch(my_node: NodeId, session_id: SessionId):
    state = State(last_event_applied_idx=1)
    chunks = _make_chunks(
        _encode(state),
        chunk_size=64,
        requester_node_id=my_node,
        session_id=session_id,
        state=state,
    )
    # Corrupt the last byte of the last chunk.
    original = chunks[-1]
    chunks[-1] = SnapshotChunk.from_data(
        data=original.data + b"\x00garbage",
        transfer_id=original.transfer_id,
        requester_node_id=original.requester_node_id,
        session_id=original.session_id,
        schema_version=original.schema_version,
        last_event_applied_idx=original.last_event_applied_idx,
        chunk_index=original.chunk_index,
        total_chunks=original.total_chunks,
        sha256_hex=original.sha256_hex,
    )

    receiver = SnapshotReceiver(my_node, session_id)
    received = None
    for chunk in chunks:
        received = receiver.ingest(chunk)
    assert received is None
