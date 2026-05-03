"""Reassembles a snapshot from a stream of `SnapshotChunk`s.

A receiver belongs to one node; it ignores chunks addressed to other
requesters and chunks from prior sessions. Once a transfer's chunks have
all been collected and the SHA-256 checks out, the snapshot is decoded into
a `State`. Concurrent transfers (for the same requester) are tolerated:
each is keyed by `transfer_id`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import final

import zstandard
from loguru import logger

from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.snapshots import SnapshotChunk, SnapshotTransferId
from exo.shared.types.state import State


@final
@dataclass
class _Assembly:
    """Partial state for one in-flight snapshot transfer."""

    total_chunks: int
    sha256_hex: str
    schema_version: int
    last_event_applied_idx: int
    chunks: dict[int, bytes] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return len(self.chunks) == self.total_chunks

    def assemble(self) -> bytes:
        return b"".join(self.chunks[i] for i in range(self.total_chunks))


@dataclass
class ReceivedSnapshot:
    last_event_applied_idx: int
    state: State


class SnapshotReceiver:
    """Filters and reassembles inbound chunks into a `ReceivedSnapshot`.

    Stateless w.r.t. delivery: callers feed `SnapshotChunk`s in via `ingest`
    and check the return value for completion.
    """

    def __init__(self, my_node_id: NodeId, session_id: SessionId) -> None:
        self._my_node_id = my_node_id
        self._session_id = session_id
        self._assemblies: dict[SnapshotTransferId, _Assembly] = {}

    def ingest(self, chunk: SnapshotChunk) -> ReceivedSnapshot | None:
        """Absorb a chunk; return the snapshot once a transfer completes.

        Returns None for partial transfers, mismatched recipients, stale
        sessions, version mismatches, or corrupt payloads.
        """
        if chunk.requester_node_id != self._my_node_id:
            return None
        if chunk.session_id != self._session_id:
            return None

        existing = self._assemblies.get(chunk.transfer_id)
        if existing is None:
            existing = _Assembly(
                total_chunks=chunk.total_chunks,
                sha256_hex=chunk.sha256_hex,
                schema_version=chunk.schema_version,
                last_event_applied_idx=chunk.last_event_applied_idx,
            )
            self._assemblies[chunk.transfer_id] = existing
        existing.chunks[chunk.chunk_index] = chunk.data

        if not existing.is_complete():
            return None

        # Transfer complete — finalise and remove from the in-flight map.
        del self._assemblies[chunk.transfer_id]
        body = existing.assemble()
        if hashlib.sha256(body).hexdigest() != existing.sha256_hex:
            logger.warning(f"Snapshot {chunk.transfer_id} failed checksum; discarding")
            return None
        try:
            decompressed = zstandard.ZstdDecompressor().decompress(body)
            state = State.model_validate_json(decompressed.decode("utf-8"))
        except (zstandard.ZstdError, ValueError) as e:
            logger.opt(exception=e).warning(
                f"Snapshot {chunk.transfer_id} could not be decoded; discarding"
            )
            return None
        if state.schema_version != existing.schema_version:
            # Should not happen — the master writes schema_version into both
            # the chunk meta and the State payload — but treat it as corrupt.
            logger.warning(
                f"Snapshot {chunk.transfer_id} schema version mismatch "
                f"(chunk={existing.schema_version}, state={state.schema_version})"
            )
            return None
        return ReceivedSnapshot(
            last_event_applied_idx=existing.last_event_applied_idx, state=state
        )
