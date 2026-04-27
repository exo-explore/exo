# type: ignore
"""Shared fixtures for exo integration tests.

These tests run against real hardware clusters managed by `eco`.
They are excluded from the default `uv run pytest` run and must be
invoked explicitly:

    uv run pytest integration_tests/ -v
"""
from __future__ import annotations

import json

import pytest

from .helpers import (
    cleanup_all_instances,
    eco_logs,
    eco_start_deploy,
    eco_stop,
    make_client,
)

# ---------------------------------------------------------------------------
# Session-scoped cluster fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def single_node_cluster():
    """Deploy a single-node cluster on s9 for the entire test session."""
    cluster = eco_start_deploy(hosts=["s9"], wait=True)
    yield cluster
    eco_stop(cluster.hosts)


@pytest.fixture(scope="session")
def two_node_cluster():
    """Deploy a 2-node cluster on s9+s10 (Thunderbolt-connected)."""
    cluster = eco_start_deploy(hosts=["s9", "s10"], wait=True)
    yield cluster
    eco_stop(cluster.hosts)


@pytest.fixture(scope="session")
def four_node_cluster():
    """Deploy a 4-node cluster on s2+s4+s9+s10."""
    cluster = eco_start_deploy(hosts=["s2", "s4", "s9", "s10"], wait=True)
    yield cluster
    eco_stop(cluster.hosts)


# ---------------------------------------------------------------------------
# Per-test instance cleanup fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup_instances_after_test(request):
    """After each test, clean up any leftover instances.

    Only cleans up fixtures the test actually used — we must NOT call
    getfixturevalue() for fixtures the test didn't request, because
    session-scoped fixtures would be instantiated (deploying a new cluster),
    which destroys the existing cluster on shared hosts.
    """
    yield
    for fixture_name in (
        "single_node_cluster",
        "two_node_cluster",
        "four_node_cluster",
    ):
        if fixture_name in request.fixturenames:
            cluster = request.getfixturevalue(fixture_name)
            client = make_client(cluster)
            cleanup_all_instances(client)
            break


# ---------------------------------------------------------------------------
# Log-on-failure hook
# ---------------------------------------------------------------------------


def pytest_runtest_makereport(item, call):
    """Attach cluster logs to the test report when a test fails."""
    if call.when != "call" or call.excinfo is None:
        return

    for fixture_name in (
        "single_node_cluster",
        "two_node_cluster",
        "four_node_cluster",
    ):
        cluster = item.funcargs.get(fixture_name)
        if cluster is not None:
            try:
                logs = eco_logs(cluster.hosts, lines=200)
                item.add_report_section(
                    "call", "Cluster Logs", json.dumps(logs, indent=2)
                )
            except Exception:
                pass
            break
