from pydantic import BaseModel, Field


class MemoryPerformanceProfile(BaseModel):
    ram_total: int
    ram_used: int
    swap_total: int
    swap_used: int


class SystemPerformanceProfile(BaseModel):
    flops_fp16: float


class NetworkInterfaceInfo(BaseModel):
    name: str
    ip_address: str
    type: str


class NodePerformanceProfile(BaseModel):
    model_id: str
    chip_id: str
    memory: MemoryPerformanceProfile
    network_interfaces: list[NetworkInterfaceInfo] = Field(default_factory=list)
    system: SystemPerformanceProfile


class ConnectionProfile(BaseModel):
    throughput: float
    latency: float
    jitter: float
