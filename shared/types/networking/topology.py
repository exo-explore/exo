from collections.abc import Sequence

from pydantic import BaseModel

from shared.types.networking.edges import (
    AddressingProtocol,
    ApplicationProtocol,
    EdgeDirection,
    EdgeId,
    EdgeInfo,
)


class Topology(BaseModel):
    edges: dict[
        EdgeId, tuple[EdgeDirection, EdgeInfo[AddressingProtocol, ApplicationProtocol]]
    ]


class NetworkState(BaseModel):
    topology: Topology
    history: Sequence[Topology]
