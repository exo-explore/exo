from enum import Enum
from typing import Generic, Mapping, Tuple, TypeVar, final

from pydantic import BaseModel, IPvAnyAddress

from shared.types.common import NewUUID, NodeId
from shared.types.graphs.common import (
    Edge,
    EdgeData,
)


class EdgeId(NewUUID):
    pass


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


class NetworkEdgeMetadata(BaseModel, Generic[AdP, ApP]): ...


@final
class NetworkEdgeType(BaseModel, Generic[AdP, ApP]):
    addressing_protocol: AdP
    application_protocol: ApP


@final
class MLXEdgeContext(
    NetworkEdgeMetadata[AddressingProtocol.IPvAny, ApplicationProtocol.MLX]
):
    source_ip: IPvAnyAddress
    sink_ip: IPvAnyAddress


class NetworkEdgeInfoType(str, Enum):
    network_profile = "network_profile"
    other = "other"


AllNetworkEdgeInfo = Tuple[NetworkEdgeInfoType.network_profile]


NetworkEdgeInfoTypeT = TypeVar(
    "NetworkEdgeInfoTypeT", bound=NetworkEdgeInfoType, covariant=True
)


class NetworkEdgeInfo(BaseModel, Generic[NetworkEdgeInfoTypeT]):
    edge_info_type: NetworkEdgeInfoTypeT


SetOfEdgeInfo = TypeVar("SetOfEdgeInfo", bound=Tuple[NetworkEdgeInfoType, ...])


class NetworkEdgeData(EdgeData[NetworkEdgeType[AdP, ApP]], Generic[AdP, ApP]):
    edge_info: Mapping[NetworkEdgeInfoType, NetworkEdgeInfo[NetworkEdgeInfoType]]
    edge_metadata: NetworkEdgeMetadata[AdP, ApP]


class NetworkEdgeProfile(NetworkEdgeInfo[NetworkEdgeInfoTypeT]):
    edge_data_transfer_rate: EdgeDataTransferRate


class NetworkEdge(Edge[NetworkEdgeType[AdP, ApP], EdgeId, NodeId]): ...
