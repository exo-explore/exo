from enum import Enum
from typing import Generic, Mapping, Tuple, TypeVar, final

from pydantic import BaseModel, IPvAnyAddress

from shared.types.common import NewUUID, NodeId
from shared.types.graphs.common import (
    Edge,
    EdgeData,
)


class DataPlaneEdgeId(NewUUID):
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


class DataPlaneEdgeMetadata(BaseModel, Generic[AdP, ApP]): ...


@final
class DataPlaneEdgeType(BaseModel, Generic[AdP, ApP]):
    addressing_protocol: AdP
    application_protocol: ApP


@final
class MLXEdgeContext(
    DataPlaneEdgeMetadata[AddressingProtocol.IPvAny, ApplicationProtocol.MLX]
):
    source_ip: IPvAnyAddress
    sink_ip: IPvAnyAddress


class DataPlaneEdgeInfoType(str, Enum):
    network_profile = "network_profile"
    other = "other"


AllDataPlaneEdgeInfo = Tuple[DataPlaneEdgeInfoType.network_profile]


DataPlaneEdgeInfoTypeT = TypeVar(
    "DataPlaneEdgeInfoTypeT", bound=DataPlaneEdgeInfoType, covariant=True
)


class DataPlaneEdgeInfo(BaseModel, Generic[DataPlaneEdgeInfoTypeT]):
    edge_info_type: DataPlaneEdgeInfoTypeT


SetOfEdgeInfo = TypeVar("SetOfEdgeInfo", bound=Tuple[DataPlaneEdgeInfoType, ...])


class DataPlaneEdgeData(EdgeData[DataPlaneEdgeType[AdP, ApP]], Generic[AdP, ApP]):
    edge_info: Mapping[DataPlaneEdgeInfoType, DataPlaneEdgeInfo[DataPlaneEdgeInfoType]]
    edge_metadata: DataPlaneEdgeMetadata[AdP, ApP]


class DataPlaneEdgeProfile(DataPlaneEdgeInfo[DataPlaneEdgeInfoTypeT]):
    edge_data_transfer_rate: EdgeDataTransferRate


class DataPlaneEdge(Edge[DataPlaneEdgeType[AdP, ApP], DataPlaneEdgeId, NodeId]): ...
