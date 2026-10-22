import time
from typing import Literal

from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel

VolunteerJobPriority = Literal["interactive", "batch_normal", "batch_low"]
VolunteerJobStatus = Literal["queued", "dispatched", "completed", "failed", "cancelled"]
VolunteerNodeStatus = Literal["healthy", "straggler", "offline"]


class VolunteerNodeProfile(FrozenModel):
    """Profile tracking a volunteer edge node's capabilities, reliability, and health."""

    node_id: str
    device_description: str = "Generic Volunteer Node"
    estimated_tps: float = 10.0
    reliability_score: float = 1.0
    status: VolunteerNodeStatus = "healthy"
    consecutive_timeouts: int = 0
    total_jobs_completed: int = 0
    last_heartbeat_timestamp: float = Field(default_factory=time.time)

    @property
    def is_available_for_dispatch(self) -> bool:
        return self.status != "offline" and self.reliability_score >= 0.2


class BatchInferenceJob(FrozenModel):
    """Work package for delay-tolerant asynchronous batch inference."""

    job_id: str
    prompt_chunks: tuple[str, ...]
    priority: VolunteerJobPriority = "batch_normal"
    timeout_seconds: float = 10.0
    redundancy_factor: int = 2
    created_at: float = Field(default_factory=time.time)

    @property
    def total_chunks(self) -> int:
        return len(self.prompt_chunks)


class GridDispatchTelemetry(FrozenModel):
    """Telemetry capturing performance and resiliency metrics of the volunteer grid."""

    telemetry_id: str
    total_jobs_dispatched: int
    jobs_completed: int
    stragglers_mitigated: int
    redundant_jobs_cancelled: int
    aggregate_tps: float
