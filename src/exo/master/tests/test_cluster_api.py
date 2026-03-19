# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for the /v1/cluster/* agent-friendly management endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import anyio
import pytest
from fastapi.testclient import TestClient

from exo.master.api import API
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeIdentity,
    NodeNetworkInfo,
    SystemPerformanceProfile,
)
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerLoading,
    RunnerReady,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata


def _make_api_with_state(state: State) -> tuple[API, TestClient]:
    """Create a minimal API instance with injected state for testing."""
    api = object.__new__(API)
    api.state = state
    api.node_id = NodeId("master-node-1")
    api.paused = False
    api.paused_ev = None  # type: ignore[assignment]
    api.command_sender = AsyncMock()

    from fastapi import FastAPI

    app = FastAPI()
    api.app = app
    api._setup_exception_handlers()

    # Register cluster routes
    app.get("/v1/cluster")(api.cluster_overview)
    app.get("/v1/cluster/health")(api.cluster_health)
    app.get("/v1/cluster/nodes")(api.cluster_nodes)
    app.get("/v1/cluster/nodes/{node_id}")(api.cluster_node_detail)
    app.get("/v1/cluster/models")(api.cluster_models)
    app.get("/v1/cluster/models/{model_id:path}/status")(api.cluster_model_status)

    client = TestClient(app)
    return api, client


def _two_node_state() -> State:
    """Build a realistic 2-node cluster state for testing."""
    topology = Topology()
    node_a = NodeId("node-atlas")
    node_b = NodeId("node-epimetheus")
    topology.add_node(node_a)
    topology.add_node(node_b)

    now = datetime.now(timezone.utc)

    model_card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3-30B-A3B-4bit"),
        storage_size=Memory.from_gb(20),
        n_layers=48,
        hidden_size=4096,
        supports_tensor=True,
        tasks=[],
    )

    runner_a = RunnerId("runner-a")
    runner_b = RunnerId("runner-b")
    instance_id = InstanceId("inst-1")
    instance = MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=ModelId("mlx-community/Qwen3-30B-A3B-4bit"),
            runner_to_shard={
                runner_a: PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=0,
                    world_size=2,
                    start_layer=0,
                    end_layer=24,
                    n_layers=48,
                ),
                runner_b: PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=1,
                    world_size=2,
                    start_layer=24,
                    end_layer=48,
                    n_layers=48,
                ),
            },
            node_to_runner={
                node_a: runner_a,
                node_b: runner_b,
            },
        ),
        hosts_by_node={node_a: [], node_b: []},
        ephemeral_port=5000,
    )

    return State(
        topology=topology,
        instances={instance_id: instance},
        runners={
            runner_a: RunnerReady(),
            runner_b: RunnerReady(),
        },
        node_identities={
            node_a: NodeIdentity(
                friendly_name="Atlas",
                chip_id="Apple M3 Ultra",
                os_version="macOS 26.2",
            ),
            node_b: NodeIdentity(
                friendly_name="Epimetheus",
                chip_id="Apple M3 Ultra",
                os_version="macOS 26.2",
            ),
        },
        node_memory={
            node_a: MemoryUsage.from_bytes(
                ram_total=512 * 1024**3,
                ram_available=340 * 1024**3,
                swap_total=0,
                swap_available=0,
            ),
            node_b: MemoryUsage.from_bytes(
                ram_total=512 * 1024**3,
                ram_available=350 * 1024**3,
                swap_total=0,
                swap_available=0,
            ),
        },
        node_disk={
            node_a: DiskUsage(
                total=Memory.from_gb(1000),
                available=Memory.from_gb(600),
            ),
            node_b: DiskUsage(
                total=Memory.from_gb(1000),
                available=Memory.from_gb(700),
            ),
        },
        node_system={
            node_a: SystemPerformanceProfile(
                gpu_usage=45.0, temp=62.0, sys_power=85.0
            ),
            node_b: SystemPerformanceProfile(
                gpu_usage=40.0, temp=58.0, sys_power=80.0
            ),
        },
        node_network={
            node_a: NodeNetworkInfo(
                interfaces=[
                    NetworkInterfaceInfo(
                        name="en0",
                        ip_address="192.168.1.10",
                        interface_type="ethernet",
                    )
                ]
            ),
            node_b: NodeNetworkInfo(
                interfaces=[
                    NetworkInterfaceInfo(
                        name="en0",
                        ip_address="192.168.1.11",
                        interface_type="ethernet",
                    )
                ]
            ),
        },
        last_seen={
            node_a: now,
            node_b: now,
        },
    )


class TestClusterHealth:
    def test_healthy_cluster(self) -> None:
        state = _two_node_state()
        _, client = _make_api_with_state(state)

        resp = client.get("/v1/cluster/health")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()

        assert data["healthy"] is True
        assert data["node_count"] == 2
        assert data["master_node_id"] == "master-node-1"
        assert "timestamp" in data

    def test_empty_cluster(self) -> None:
        _, client = _make_api_with_state(State())

        resp = client.get("/v1/cluster/health")
        data: dict[str, Any] = resp.json()

        assert data["healthy"] is False
        assert data["node_count"] == 0


class TestClusterOverview:
    def test_overview_structure(self) -> None:
        state = _two_node_state()
        _, client = _make_api_with_state(state)

        resp = client.get("/v1/cluster")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()

        # Top-level summary
        assert data["node_count"] == 2
        assert data["total_ram_gb"] == 1024.0
        assert data["loaded_model_count"] == 1
        assert data["master_node_id"] == "master-node-1"

        # Nodes
        assert len(data["nodes"]) == 2
        atlas = next(n for n in data["nodes"] if n["friendly_name"] == "Atlas")
        assert atlas["chip"] == "Apple M3 Ultra"
        assert atlas["ram_total_gb"] == 512.0
        assert atlas["status"] == "online"
        assert "Qwen3-30B-A3B-4bit" in atlas["loaded_models"]

        # Models
        assert len(data["models"]) == 1
        model = data["models"][0]
        assert model["model_name"] == "Qwen3-30B-A3B-4bit"
        assert model["ready"] is True
        assert len(model["nodes"]) == 2
        assert model["storage_size_gb"] == 20.0

    def test_overview_memory_math(self) -> None:
        state = _two_node_state()
        _, client = _make_api_with_state(state)

        data: dict[str, Any] = client.get("/v1/cluster").json()

        # 340 + 350 = 690 available, 1024 - 690 = 334 used
        assert data["available_ram_gb"] == 690.0
        assert data["used_ram_gb"] == 334.0
        assert data["ram_used_percent"] == pytest.approx(32.6, abs=0.1)


class TestClusterNodes:
    def test_list_nodes(self) -> None:
        state = _two_node_state()
        _, client = _make_api_with_state(state)

        resp = client.get("/v1/cluster/nodes")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()

        assert data["node_count"] == 2
        assert len(data["nodes"]) == 2

    def test_node_detail(self) -> None:
        state = _two_node_state()
        _, client = _make_api_with_state(state)

        resp = client.get("/v1/cluster/nodes/node-atlas")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()

        assert data["friendly_name"] == "Atlas"
        assert data["ram_total_gb"] == 512.0
        assert data["gpu_usage_percent"] == 45.0
        assert "192.168.1.10" in data["ip_addresses"]

    def test_node_not_found(self) -> None:
        state = _two_node_state()
        _, client = _make_api_with_state(state)

        resp = client.get("/v1/cluster/nodes/nonexistent")
        assert resp.status_code == 404
        data: dict[str, Any] = resp.json()
        # Error should list available nodes
        assert "node-atlas" in data["error"]["message"] or "Available nodes" in data["error"]["message"]


class TestClusterModels:
    def test_list_loaded_models(self) -> None:
        state = _two_node_state()
        _, client = _make_api_with_state(state)

        resp = client.get("/v1/cluster/models")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()

        assert len(data["loaded"]) == 1
        model = data["loaded"][0]
        assert model["model_id"] == "mlx-community/Qwen3-30B-A3B-4bit"
        assert model["model_name"] == "Qwen3-30B-A3B-4bit"
        assert model["sharding"] == "pipeline"
        assert model["instance_type"] == "MlxRing"
        assert model["ready"] is True
        assert "Atlas" in model["node_names"]
        assert "Epimetheus" in model["node_names"]

    def test_empty_cluster_models(self) -> None:
        _, client = _make_api_with_state(State())

        data: dict[str, Any] = client.get("/v1/cluster/models").json()
        assert len(data["loaded"]) == 0
        assert len(data["downloading"]) == 0


class TestClusterModelStatus:
    def test_loaded_model_status(self) -> None:
        state = _two_node_state()
        _, client = _make_api_with_state(state)

        # By full model ID
        resp = client.get("/v1/cluster/models/mlx-community/Qwen3-30B-A3B-4bit/status")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()

        assert data["found"] is True
        assert data["ready"] is True
        assert data["status"] == "ready"
        assert data["model_id"] == "mlx-community/Qwen3-30B-A3B-4bit"
        assert "Atlas" in data["nodes"]
        assert data["instance_id"] is not None

    def test_loaded_model_status_by_short_name(self) -> None:
        state = _two_node_state()
        _, client = _make_api_with_state(state)

        resp = client.get("/v1/cluster/models/Qwen3-30B-A3B-4bit/status")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()

        assert data["found"] is True
        assert data["ready"] is True

    def test_not_loaded_model_status(self) -> None:
        state = _two_node_state()
        _, client = _make_api_with_state(state)

        resp = client.get("/v1/cluster/models/nonexistent-model/status")
        assert resp.status_code == 200  # Not a 404 — this is a status check
        data: dict[str, Any] = resp.json()

        assert data["found"] is False
        assert data["ready"] is False
        assert data["status"] == "not_loaded"

    def test_loading_model_shows_progress(self) -> None:
        """When a runner is in Loading state, status should show layer progress."""
        state = _two_node_state()
        # Replace one runner with a Loading state
        runner_a = list(state.runners.keys())[0]
        state = state.model_copy(
            update={
                "runners": {
                    **state.runners,
                    runner_a: RunnerLoading(layers_loaded=12, total_layers=48),
                }
            }
        )
        _, client = _make_api_with_state(state)

        data: dict[str, Any] = client.get(
            "/v1/cluster/models/Qwen3-30B-A3B-4bit/status"
        ).json()

        assert data["found"] is True
        assert data["ready"] is False
        assert data["status"] == "loading"
        assert "12/48" in data["progress"]
        assert "25%" in data["progress"]
