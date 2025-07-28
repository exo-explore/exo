from pydantic import BaseModel, Field


class MemoryPerformanceProfile(BaseModel):
    ram_total: int
    ram_available: int
    swap_total: int
    swap_available: int


class SystemPerformanceProfile(BaseModel):
    flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0
    ane_power: float = 0.0


class NetworkInterfaceInfo(BaseModel):
    name: str
    ip_address: str
    type: str


class NodePerformanceProfile(BaseModel):
    model_id: str
    chip_id: str
    friendly_name: str
    memory: MemoryPerformanceProfile
    network_interfaces: list[NetworkInterfaceInfo] = Field(default_factory=list)
    system: SystemPerformanceProfile


class ConnectionProfile(BaseModel):
    throughput: float
    latency: float
    jitter: float
