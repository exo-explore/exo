"""Agent-friendly cluster management response types.

These types power the ``/v1/cluster/*`` endpoints, designed for programmatic
cluster management by AI agents, CLI tools, and automation scripts.  Every
response is flat, self-describing, and includes units in field names so
consumers never have to guess.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.worker.instances import InstanceId


# ---------------------------------------------------------------------------
# Node summary
# ---------------------------------------------------------------------------

class ClusterNodeSummary(BaseModel):
    """One node in the cluster, with everything an agent needs at a glance."""

    node_id: str
    friendly_name: str = "Unknown"
    chip: str = "Unknown"
    os_version: str = "Unknown"

    # Memory (GB, rounded to 1 decimal)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_used_percent: float = 0.0

    # Disk
    disk_total_gb: float = 0.0
    disk_available_gb: float = 0.0

    # Performance
    gpu_usage_percent: float = 0.0
    cpu_p_usage_percent: float = 0.0
    cpu_e_usage_percent: float = 0.0
    temperature_c: float = 0.0
    power_watts: float = 0.0

    # Network
    ip_addresses: list[str] = Field(default_factory=list)
    connection_types: list[str] = Field(default_factory=list)

    # Thunderbolt / RDMA
    has_thunderbolt: bool = False
    rdma_enabled: bool = False

    # What's loaded on this node
    loaded_models: list[str] = Field(default_factory=list)

    # Status
    status: Literal["online", "stale", "unknown"] = "unknown"
    seconds_since_seen: float | None = None


# ---------------------------------------------------------------------------
# Loaded model summary
# ---------------------------------------------------------------------------

class ClusterModelSummary(BaseModel):
    """A model instance currently loaded on the cluster."""

    instance_id: str
    model_id: str
    model_name: str  # Short name (e.g. "Qwen3-30B-A3B")
    sharding: str  # "pipeline" or "tensor"
    instance_type: str  # "MlxRing" or "MlxJaccl"
    nodes: list[str]  # Node IDs hosting this model
    node_names: list[str]  # Friendly names of nodes
    storage_size_gb: float = 0.0

    # Runner status per node
    runner_statuses: dict[str, str] = Field(default_factory=dict)

    # Overall readiness
    ready: bool = False
    status: str = "unknown"  # "loading", "ready", "running", "failed", etc.


# ---------------------------------------------------------------------------
# Active task summary
# ---------------------------------------------------------------------------

class ClusterTaskSummary(BaseModel):
    """An in-flight task on the cluster."""

    task_id: str
    task_type: str  # "text_generation", "image_generation", etc.
    status: str  # "pending", "running", "complete", "failed"
    model_id: str = ""
    instance_id: str = ""


# ---------------------------------------------------------------------------
# Download summary
# ---------------------------------------------------------------------------

class ClusterDownloadSummary(BaseModel):
    """A model download in progress or completed."""

    node_id: str
    node_name: str
    model_id: str
    status: Literal["pending", "downloading", "completed", "failed"]
    downloaded_gb: float = 0.0
    total_gb: float = 0.0
    progress_percent: float = 0.0
    speed_mb_s: float = 0.0
    eta_seconds: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Cluster overview (the "one call" response)
# ---------------------------------------------------------------------------

class ClusterHealthResponse(BaseModel):
    """Quick health check — is the cluster alive?"""

    healthy: bool
    node_count: int
    master_node_id: str
    uptime_seconds: float | None = None
    timestamp: float = Field(default_factory=time.time)


class ClusterOverviewResponse(BaseModel):
    """Everything an agent needs in one call.

    Designed so that a single ``GET /v1/cluster`` gives full situational
    awareness without parsing raw state or cross-referencing endpoints.
    """

    # Top-level summary
    node_count: int = 0
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    used_ram_gb: float = 0.0
    ram_used_percent: float = 0.0
    loaded_model_count: int = 0
    active_task_count: int = 0
    active_download_count: int = 0

    # Details
    nodes: list[ClusterNodeSummary] = Field(default_factory=list)
    models: list[ClusterModelSummary] = Field(default_factory=list)
    tasks: list[ClusterTaskSummary] = Field(default_factory=list)
    downloads: list[ClusterDownloadSummary] = Field(default_factory=list)

    # Metadata
    master_node_id: str = ""
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Action requests (simplified agent-friendly versions)
# ---------------------------------------------------------------------------

class LoadModelRequest(BaseModel):
    """Load a model by name — the cluster figures out placement."""

    model_id: str
    min_nodes: int = 1
    preferred_sharding: Literal["pipeline", "tensor", "auto"] = "auto"


class LoadModelResponse(BaseModel):
    """Response after requesting a model load."""

    message: str
    command_id: str
    model_id: str
    instance_id: str | None = None
    placement: dict[str, str] = Field(default_factory=dict)  # node_id -> role
    estimated_memory_gb: float = 0.0


class UnloadModelResponse(BaseModel):
    """Response after requesting a model unload."""

    message: str
    command_id: str
    model_id: str
    instance_id: str
    freed_memory_gb: float = 0.0


class SwapModelRequest(BaseModel):
    """Atomically swap one model for another.

    Unloads ``unload_model_id`` then loads ``load_model_id`` in a single call.
    Ideal for day/night model rotation scripts.
    """

    unload_model_id: str
    load_model_id: str
    min_nodes: int = 1
    preferred_sharding: Literal["pipeline", "tensor", "auto"] = "auto"


class SwapModelResponse(BaseModel):
    """Response after a model swap request."""

    message: str
    unload_command_id: str
    load_command_id: str
    unloaded_model: str
    loaded_model: str
    freed_memory_gb: float = 0.0
    estimated_load_memory_gb: float = 0.0


class ModelStatusResponse(BaseModel):
    """Polling-friendly status for a single model by name.

    Hit this in a loop to wait for a model to become ready after loading,
    or to confirm it's fully unloaded after deletion.
    """

    model_id: str
    found: bool = False
    status: str = "not_loaded"  # not_loaded | downloading | loading | ready | running | failed
    ready: bool = False
    progress: str | None = None  # Human-readable progress string
    nodes: list[str] = Field(default_factory=list)
    instance_id: str | None = None


class ClusterNodesResponse(BaseModel):
    """List of all nodes in the cluster."""

    node_count: int
    nodes: list[ClusterNodeSummary]
    timestamp: float = Field(default_factory=time.time)


class ClusterModelsResponse(BaseModel):
    """All models: loaded, downloading, and available."""

    loaded: list[ClusterModelSummary] = Field(default_factory=list)
    downloading: list[ClusterDownloadSummary] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)
