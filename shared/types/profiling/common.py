from typing import Annotated, Literal, Coroutine, Generic, TypeVar
from enum import Enum
from abc import ABC
from pydantic import BaseModel, Field, TypeAdapter


class ProfiledResourceName(str, Enum):
    memory = 'memory'   
    system = 'system'

ProfiledResourceT = TypeVar(name='ProfiledResourceT', bound=ProfiledResourceName)

class BasePerformanceProfile(BaseModel, Generic[ProfiledResourceT]):
    """
    Details a single resource (or resource type) that is being monitored by the resource monitor.
    """
    pass

class MemoryPerformanceProfile(BasePerformanceProfile[ProfiledResourceName.memory]):
    resource_name: Literal[ProfiledResourceName.memory] = Field(
        default=ProfiledResourceName.memory, frozen=True
    )
    ram_total: int
    ram_used: int
    swap_total: int
    swap_used: int

class NetworkInterfaceInfo(BaseModel):
    name: str
    ip_address: str
    type: str

class SystemPerformanceProfile(BasePerformanceProfile[ProfiledResourceName.system]):
    resource_name: Literal[ProfiledResourceName.system] = Field(
        default=ProfiledResourceName.system, frozen=True
    )
    model_id: str
    chip_id: str
    memory: int
    network_interfaces: list[NetworkInterfaceInfo] = Field(default_factory=list)

NodePerformanceProfile = Annotated[
    MemoryPerformanceProfile | SystemPerformanceProfile,
    Field(discriminator="resource_name")
]

NodePerformanceProfileTypeAdapter: TypeAdapter[NodePerformanceProfile] = TypeAdapter(NodePerformanceProfile)