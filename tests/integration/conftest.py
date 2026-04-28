# type: ignore
"""Shared fixtures for exo integration tests.

These tests run against real hardware clusters managed by `eco`.
They are excluded from the default `uv run pytest` run and must be
invoked explicitly:

    uv run pytest tests/integration/ -v

To override host selection (instead of constraint-based reservation):

    uv run pytest tests/integration/ -v --hosts s2,s4,s9,s10
"""

from __future__ import annotations

import json

import pytest

from .helpers import (
    _ECO_USER,
    cleanup_all_instances,
    eco_logs,
    eco_start_deploy,
    eco_stop,
    make_client,
)


def pytest_addoption(parser):
    parser.addoption(
        "--hosts",
        default=None,
        help="Comma-separated list of hosts to use (e.g. s2,s4,s9,s10). "
        "Overrides constraint-based reservation. Hosts are allocated "
        "to fixtures in order: first N for N-node clusters.",
    )


def pytest_report_header(config):
    """Show the eco user and hosts for this test session."""
    hosts = config.getoption("--hosts")
    lines = [f"eco user: {_ECO_USER}"]
    if hosts:
        lines.append(f"hosts override: {hosts}")
    return lines


@pytest.fixture(scope="session")
def _host_pool(request):
    """Parse --hosts into a list, or None for constraint-based reservation."""
    raw = request.config.getoption("--hosts")
    if raw:
        return [h.strip() for h in raw.split(",") if h.strip()]
    return None


# ---------------------------------------------------------------------------
# Session-scoped cluster fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def single_node_cluster(_host_pool):
    """Deploy a single-node cluster."""
    if _host_pool:
        cluster = eco_start_deploy(hosts=_host_pool[:1], wait=True)
    else:
        cluster = eco_start_deploy(count=1, wait=True)
    yield cluster
    eco_stop(cluster.hosts)


@pytest.fixture(scope="session")
def two_node_cluster(_host_pool):
    """Deploy a 2-node Thunderbolt-connected cluster."""
    if _host_pool:
        cluster = eco_start_deploy(hosts=_host_pool[:2], wait=True)
    else:
        cluster = eco_start_deploy(count=2, thunderbolt=True, wait=True)
    yield cluster
    eco_stop(cluster.hosts)


@pytest.fixture(scope="session")
def four_node_cluster(_host_pool):
    """Deploy a 4-node Thunderbolt-connected cluster."""
    if _host_pool:
        cluster = eco_start_deploy(hosts=_host_pool[:4], wait=True)
    else:
        cluster = eco_start_deploy(count=4, thunderbolt=True, wait=True)
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
