from enum import Enum
from typing import Annotated, Literal, TypeVar, Union, final

from pydantic import BaseModel, Field, IPvAnyAddress, TypeAdapter

from shared.types.common import NewUUID, NodeId
from shared.types.graphs.common import Edge


class DataPlaneEdgeId(NewUUID):
    pass


class AddressingProtocol(str, Enum):
    IPvAnyAddress = "IPvAnyAddress"


class ApplicationProtocol(str, Enum):
    MLX = "MLX"


AdP = TypeVar("AdP", bound=AddressingProtocol)
ApP = TypeVar("ApP", bound=ApplicationProtocol)


@final
class DataPlaneEdgeProfile(BaseModel):
    throughput: float
    latency: float
    jitter: float


class CommonDataPlaneEdgeData(BaseModel):
    edge_data_transfer_rate: DataPlaneEdgeProfile | None = None


class MlxEdgeMetadata(BaseModel):
    source_ip: IPvAnyAddress
    sink_ip: IPvAnyAddress


class BaseDataPlaneEdgeData[AdP: AddressingProtocol, ApP: ApplicationProtocol](
    BaseModel
):
    addressing_protocol: AdP
    application_protocol: ApP
    common_data: CommonDataPlaneEdgeData


class MlxEdge(
    BaseDataPlaneEdgeData[AddressingProtocol.IPvAnyAddress, ApplicationProtocol.MLX]
):
    addressing_protocol: Literal[AddressingProtocol.IPvAnyAddress] = (
        AddressingProtocol.IPvAnyAddress
    )
    application_protocol: Literal[ApplicationProtocol.MLX] = ApplicationProtocol.MLX
    mlx_metadata: MlxEdgeMetadata


DataPlaneEdgeData = Union[MlxEdge]

_DataPlaneEdgeData = Annotated[
    DataPlaneEdgeData,
    Field(discriminator="addressing_protocol"),
]
DataPlaneEdgeAdapter: TypeAdapter[DataPlaneEdgeData] = TypeAdapter(_DataPlaneEdgeData)

DataPlaneEdge = Edge[DataPlaneEdgeData, DataPlaneEdgeId, NodeId]
