from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Generic, NamedTuple, TypeVar, final
from uuid import UUID

from pydantic import BaseModel, IPvAnyAddress, TypeAdapter
from pydantic.types import UuidVersion

from shared.types.common import NodeId

_EdgeId = Annotated[UUID, UuidVersion(4)]
EdgeId = type("EdgeId", (UUID,), {})
EdgeIdParser: TypeAdapter[EdgeId] = TypeAdapter(_EdgeId)


@final
class EdgeDataTransferRate(BaseModel):
    throughput: float
    latency: float
    jitter: float


class AddressingProtocol(str, Enum):
    IPvAny = "IPvAny"


class ApplicationProtocol(str, Enum):
    MLX = "MLX"


TE = TypeVar("TE", bound=AddressingProtocol)
TF = TypeVar("TF", bound=ApplicationProtocol)


@final
class EdgeType(BaseModel, Generic[TE, TF]):
    addressing_protocol: TE
    application_protocol: TF


@final
class EdgeDirection(NamedTuple):
    source: NodeId
    sink: NodeId


@dataclass
class EdgeMetadata(BaseModel, Generic[TE, TF]): ...


@final
@dataclass
class MLXEdgeContext(EdgeMetadata[AddressingProtocol.IPvAny, ApplicationProtocol.MLX]):
    source_ip: IPvAnyAddress
    sink_ip: IPvAnyAddress


@final
class EdgeInfo(BaseModel, Generic[TE, TF]):
    edge_type: EdgeType[TE, TF]
    edge_data_transfer_rate: EdgeDataTransferRate
    edge_metadata: EdgeMetadata[TE, TF]


@final
class DirectedEdge(BaseModel, Generic[TE, TF]):
    edge_direction: EdgeDirection
    edge_identifier: EdgeId
    edge_info: EdgeInfo[TE, TF]


"""
an_edge: DirectedEdge[Literal[AddressingProtocol.IPvAny], Literal[ApplicationProtocol.MLX]] = DirectedEdge(
    edge_identifier=UUID(),
    edge_direction=EdgeDirection(source=NodeId("1"), sink=NodeId("2")),
    edge_info=EdgeInfo(
        edge_type=EdgeType(
            addressing_protocol=AddressingProtocol.ipv4,
            application_protocol=ApplicationProtocol.mlx
        ),
        edge_data_transfer_rate=EdgeDataTransferRate(throughput=1000, latency=0.1, jitter=0.01),
        edge_metadata=MLXEdgeContext(source_ip=IpV4Addr("192.168.1.1"), sink_ip=IpV4Addr("192.168.1.2"))
    )
)
"""
