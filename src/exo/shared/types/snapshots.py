"""Wire types for snapshot transfer between master and a joining node.

Snapshots can be tens of MB; the gossipsub message ceiling is around 1 MB.
We slice the compressed snapshot body into chunks and publish each chunk on
the SNAPSHOT_RESPONSES topic. The receiver collects chunks for its own
`requester_node_id`, validates the SHA-256 of the reassembled body, and
materialises the State.
"""

import base64

from exo.shared.types.common import Id, NodeId, SessionId
from exo.utils.pydantic_ext import FrozenModel


class SnapshotTransferId(Id):
    """Identifies a single snapshot transfer (one master response to one
    `RequestSnapshot`). Distinct transfers may interleave; the id lets
    receivers keep them apart."""


class SnapshotChunk(FrozenModel):
    """One slice of a snapshot in flight.

    `data_b64` carries a base64-encoded slice of the zstd-compressed JSON
    dump of State. Concatenating the *decoded* bytes of all chunks for a
    `transfer_id` in order of `chunk_index` yields the full compressed
    body; `sha256_hex` is the SHA-256 of that decoded blob.

    We use base64 explicitly because the topic layer JSON-encodes messages,
    and JSON can't carry raw binary. Helpers `from_data` / `data` keep the
    base64 detail at the boundaries.
    """

    transfer_id: SnapshotTransferId
    requester_node_id: NodeId
    session_id: SessionId
    schema_version: int
    last_event_applied_idx: int
    chunk_index: int
    total_chunks: int
    sha256_hex: str
    data_b64: str

    @classmethod
    def from_data(cls, *, data: bytes, **kwargs: object) -> "SnapshotChunk":
        return cls(data_b64=base64.b64encode(data).decode("ascii"), **kwargs)  # pyright: ignore[reportArgumentType]

    @property
    def data(self) -> bytes:
        return base64.b64decode(self.data_b64)


__all__ = ["SnapshotChunk", "SnapshotTransferId"]
