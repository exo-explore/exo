from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Generic, NamedTuple, TypeVar, final
from uuid import UUID

from pydantic import AfterValidator, BaseModel, IPvAnyAddress, TypeAdapter
from pydantic.types import UuidVersion

from shared.types.common import NodeId

_EdgeId = Annotated[UUID, UuidVersion(4)]
EdgeId = type("EdgeId", (UUID,), {})
EdgeIdParser: TypeAdapter[EdgeId] = TypeAdapter(_EdgeId)


class AddressingProtocol(str, Enum):
    IPvAny = "IPvAny"


class ApplicationProtocol(str, Enum):
    MLX = "MLX"


AdP = TypeVar("AdP", bound=AddressingProtocol)
ApP = TypeVar("ApP", bound=ApplicationProtocol)


@final
class EdgeDataTransferRate(BaseModel):
    throughput: float
    latency: float
    jitter: float


class EdgeMetadata(BaseModel, Generic[AdP, ApP]): ...


@final
class EdgeType(BaseModel, Generic[AdP, ApP]):
    addressing_protocol: AdP
    application_protocol: ApP


@final
class EdgeDirection(NamedTuple):
    source: NodeId
    sink: NodeId


@final
class MLXEdgeContext(EdgeMetadata[AddressingProtocol.IPvAny, ApplicationProtocol.MLX]):
    source_ip: IPvAnyAddress
    sink_ip: IPvAnyAddress


class EdgeDataType(str, Enum):
    DISCOVERED = "discovered"
    PROFILED = "profiled"
    UNKNOWN = "unknown"


EdgeDataTypeT = TypeVar("EdgeDataTypeT", bound=EdgeDataType)


class EdgeData(BaseModel, Generic[EdgeDataTypeT]):
    edge_data_type: EdgeDataTypeT


class EdgeProfile(EdgeData[EdgeDataType.PROFILED]):
    edge_data_transfer_rate: EdgeDataTransferRate


def validate_mapping(
    edge_data: Mapping[EdgeDataType, EdgeData[EdgeDataType]],
) -> Mapping[EdgeDataType, EdgeData[EdgeDataType]]:
    """Validates that each EdgeData value has an edge_data_type matching its key."""
    for key, value in edge_data.items():
        if key != value.edge_data_type:
            raise ValueError(
                f"Edge Data Type Mismatch: key {key} != value {value.edge_data_type}"
            )
    return edge_data


class Edge(BaseModel, Generic[AdP, ApP, EdgeDataTypeT]):
    edge_type: EdgeType[AdP, ApP]
    edge_direction: EdgeDirection
    edge_data: Annotated[
        Mapping[EdgeDataType, EdgeData[EdgeDataType]], AfterValidator(validate_mapping)
    ]
    edge_metadata: EdgeMetadata[AdP, ApP]


"""
an_edge: UniqueEdge[Literal[AddressingProtocol.IPvAny], Literal[ApplicationProtocol.MLX]] = UniqueEdge(
    edge_identifier=EdgeId(UUID().hex),
    edge_info=ProfiledEdge(
        edge_direction=EdgeDirection(source=NodeId("1"), sink=NodeId("2")),
        edge_type=EdgeType(
            addressing_protocol=AddressingProtocol.IPvAny,
            application_protocol=ApplicationProtocol.MLX
        ),
        edge_data=EdgeData(
            edge_data_transfer_rate=EdgeDataTransferRate(throughput=1000, latency=0.1, jitter=0.01)
        ),
        edge_metadata=MLXEdgeContext(source_ip=IPv4Address("192.168.1.1"), sink_ip=IPv4Address("192.168.1.2"))
    )
)
"""
