from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel

from shared.types.networking.edges import (
    AddressingProtocol,
    ApplicationProtocol,
    Edge,
    EdgeDataType,
    EdgeId,
)


class Topology(BaseModel):
    edges: Mapping[
        EdgeId,
        Edge[AddressingProtocol, ApplicationProtocol, Literal[EdgeDataType.DISCOVERED]],
    ]


class EdgeMap(BaseModel):
    edges: Mapping[EdgeId, Edge[AddressingProtocol, ApplicationProtocol, EdgeDataType]]


class NetworkState(BaseModel):
    topology: Topology
    history: Sequence[Topology]
