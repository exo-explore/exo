# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for exo-cli argument parsing and output formatting."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from exo.cli.client import ExoClient
from exo.cli.format import (
    format_health,
    format_models,
    format_node_detail,
    format_nodes,
    format_overview,
)
from exo.cli.main import build_parser, cmd_health, cmd_models, cmd_nodes, cmd_status


# ---------------------------------------------------------------------------
# Fixtures — realistic API responses
# ---------------------------------------------------------------------------

HEALTH_RESPONSE: dict[str, Any] = {
    "healthy": True,
    "node_count": 2,
    "master_node_id": "abc-123",
    "timestamp": 1710000000.0,
}

OVERVIEW_RESPONSE: dict[str, Any] = {
    "node_count": 2,
    "total_ram_gb": 1024.0,
    "available_ram_gb": 690.0,
    "used_ram_gb": 334.0,
    "ram_used_percent": 32.6,
    "loaded_model_count": 1,
    "active_task_count": 0,
    "active_download_count": 0,
    "nodes": [
        {
            "node_id": "node-atlas",
            "friendly_name": "Atlas",
            "chip": "Apple M3 Ultra",
            "os_version": "macOS 26.2",
            "ram_total_gb": 512.0,
            "ram_available_gb": 340.0,
            "ram_used_gb": 172.0,
            "ram_used_percent": 33.6,
            "disk_total_gb": 1000.0,
            "disk_available_gb": 600.0,
            "gpu_usage_percent": 45.0,
            "cpu_p_usage_percent": 10.0,
            "cpu_e_usage_percent": 5.0,
            "temperature_c": 62.0,
            "power_watts": 85.0,
            "ip_addresses": ["192.168.1.10"],
            "connection_types": ["ethernet"],
            "has_thunderbolt": True,
            "rdma_enabled": False,
            "loaded_models": ["Qwen3-30B-A3B-4bit"],
            "status": "online",
            "seconds_since_seen": 2.1,
        },
        {
            "node_id": "node-epimetheus",
            "friendly_name": "Epimetheus",
            "chip": "Apple M3 Ultra",
            "os_version": "macOS 26.2",
            "ram_total_gb": 512.0,
            "ram_available_gb": 350.0,
            "ram_used_gb": 162.0,
            "ram_used_percent": 31.6,
            "disk_total_gb": 1000.0,
            "disk_available_gb": 700.0,
            "gpu_usage_percent": 40.0,
            "cpu_p_usage_percent": 8.0,
            "cpu_e_usage_percent": 3.0,
            "temperature_c": 58.0,
            "power_watts": 80.0,
            "ip_addresses": ["192.168.1.11"],
            "connection_types": ["ethernet"],
            "has_thunderbolt": True,
            "rdma_enabled": False,
            "loaded_models": ["Qwen3-30B-A3B-4bit"],
            "status": "online",
            "seconds_since_seen": 1.8,
        },
    ],
    "models": [
        {
            "instance_id": "inst-1",
            "model_id": "mlx-community/Qwen3-30B-A3B-4bit",
            "model_name": "Qwen3-30B-A3B-4bit",
            "sharding": "pipeline",
            "instance_type": "MlxRing",
            "nodes": ["node-atlas", "node-epimetheus"],
            "node_names": ["Atlas", "Epimetheus"],
            "storage_size_gb": 20.0,
            "runner_statuses": {},
            "ready": True,
            "status": "ready",
        }
    ],
    "tasks": [],
    "downloads": [],
    "master_node_id": "abc-123",
    "timestamp": 1710000000.0,
}

NODES_RESPONSE: dict[str, Any] = {
    "node_count": 2,
    "nodes": OVERVIEW_RESPONSE["nodes"],
    "timestamp": 1710000000.0,
}

MODELS_RESPONSE: dict[str, Any] = {
    "loaded": OVERVIEW_RESPONSE["models"],
    "downloading": [],
    "timestamp": 1710000000.0,
}


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_status_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_health_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["health"])
        assert args.command == "health"

    def test_nodes_list(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["nodes"])
        assert args.command == "nodes"
        assert args.node_id is None

    def test_nodes_detail(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["nodes", "node-atlas"])
        assert args.node_id == "node-atlas"

    def test_models_list(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["models"])
        assert args.command == "models"

    def test_models_load(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["models", "load", "mlx-community/Qwen3-30B"])
        assert args.models_action == "load"
        assert args.model_name == "mlx-community/Qwen3-30B"
        assert args.wait is False

    def test_models_load_wait(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["models", "load", "--wait", "mlx-community/Qwen3-30B"])
        assert args.wait is True

    def test_models_swap(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["models", "swap", "old-model", "new-model"])
        assert args.models_action == "swap"
        assert args.unload_name == "old-model"
        assert args.load_name == "new-model"

    def test_models_swap_wait_with_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "models", "swap", "--wait", "--min-nodes", "2",
            "--sharding", "tensor", "old", "new"
        ])
        assert args.wait is True
        assert args.min_nodes == 2
        assert args.sharding == "tensor"

    def test_global_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--host", "atlas", "--port", "9999", "--json", "status"])
        assert args.host == "atlas"
        assert args.port == 9999
        assert args.json is True

    def test_models_unload(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["models", "unload", "Qwen3-30B"])
        assert args.models_action == "unload"
        assert args.model_name == "Qwen3-30B"


# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------

class TestFormatters:
    def test_format_health_healthy(self) -> None:
        out = format_health(HEALTH_RESPONSE)
        assert "healthy" in out
        assert "2" in out

    def test_format_health_unhealthy(self) -> None:
        out = format_health({**HEALTH_RESPONSE, "healthy": False})
        assert "unhealthy" in out

    def test_format_overview_structure(self) -> None:
        out = format_overview(OVERVIEW_RESPONSE)
        assert "2 nodes" in out
        assert "Atlas" in out
        assert "Qwen3-30B-A3B-4bit" in out
        assert "1024.0" in out

    def test_format_nodes(self) -> None:
        out = format_nodes(NODES_RESPONSE)
        assert "2 node(s)" in out
        assert "Atlas" in out
        assert "Epimetheus" in out
        assert "M3 Ultra" in out

    def test_format_node_detail(self) -> None:
        out = format_node_detail(OVERVIEW_RESPONSE["nodes"][0])
        assert "Atlas" in out
        assert "512.0" in out
        assert "192.168.1.10" in out
        assert "Thunderbolt" in out

    def test_format_models(self) -> None:
        out = format_models(MODELS_RESPONSE)
        assert "Qwen3-30B-A3B-4bit" in out
        assert "Atlas" in out
        assert "ready" in out

    def test_format_models_empty(self) -> None:
        out = format_models({"loaded": [], "downloading": []})
        assert "No models loaded" in out


# ---------------------------------------------------------------------------
# Client URL construction
# ---------------------------------------------------------------------------

class TestClientURLs:
    def test_default_base_url(self) -> None:
        c = ExoClient()
        assert c.base_url == "http://localhost:52415"

    def test_custom_host_port(self) -> None:
        c = ExoClient(host="atlas.miller.lan", port=9999)
        assert c.base_url == "http://atlas.miller.lan:9999"
