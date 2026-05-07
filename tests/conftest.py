# type: ignore
"""Pytest configuration for marker-driven exo integration tests.

Test authors declare requirements via markers:

    @pytest.mark.cluster(count=2, thunderbolt='a2a')
    @pytest.mark.instance('mlx-community/Llama-3.2-1B-Instruct-4bit',
                          sharding='tensor', comm='jaccl')
    def test_jaccl_inference(session):
        resp = session.chat('What is 2+2?')
        assert '4' in resp

Clusters are cached by `ClusterSpec`; tests with the same cluster_spec
share a deployment. Each test places its own instance (matching its
`@pytest.mark.instance`), and instances are cleaned up after the test.

Run with:
    uv run pytest tests/ -v
    uv run pytest tests/ -v --hosts s2,s4,s9,s10
"""

from __future__ import annotations

import contextlib
import json

import pytest
from exo_tools.cluster import ClusterInfo, EcoSession
from exo_tools.harness import cleanup_all_instances, place_instance

from .framework import (
    ClusterSpec,
    Session,
    parse_cluster_marker,
    parse_instance_marker,
)

# Single eco session for the entire test process.
eco = EcoSession(user_prefix="test")

# Cluster cache keyed by ClusterSpec — tests with the same spec share a deployment.
# Cleared at session teardown.
_cluster_cache: dict[ClusterSpec, ClusterInfo] = {}


def pytest_addoption(parser):
    parser.addoption(
        "--hosts",
        default=None,
        help="Comma-separated list of hosts (e.g. s2,s4,s9,s10). "
        "Overrides constraint-based reservation.",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "cluster(count=N, thunderbolt=Thunderbolt|None, min_memory=GB, chip=PATTERN): "
        "declare cluster requirements for a test",
    )
    config.addinivalue_line(
        "markers",
        "instance(model_id, sharding=Sharding, comm=Comm, min_nodes=N): "
        "declare instance placement for a test",
    )


def pytest_report_header(config):
    """Show the eco user and hosts for this test session."""
    hosts = config.getoption("--hosts")
    lines = [f"eco user: {eco.user}"]
    if hosts:
        lines.append(f"hosts override: {hosts}")
    return lines


@pytest.fixture(scope="session")
def _host_pool(request) -> list[str] | None:
    raw = request.config.getoption("--hosts")
    if raw:
        return [h.strip() for h in raw.split(",") if h.strip()]
    return None


@pytest.fixture
def session(request, _host_pool) -> Session:
    """Per-test fixture providing a Session matching the test's markers.

    Reads @pytest.mark.cluster and @pytest.mark.instance from the test, deploys
    a matching cluster (cached across tests with the same spec), places the
    model, and yields a Session for the test to interact with. Cleans up the
    instance after the test, and invalidates the cluster cache if the test
    left nodes disconnected.
    """
    cluster_marker = request.node.get_closest_marker("cluster")
    instance_marker = request.node.get_closest_marker("instance")

    cluster_spec = parse_cluster_marker(cluster_marker)
    instance_spec = parse_instance_marker(instance_marker)

    # Deploy or reuse a cluster matching the spec
    cluster = _cluster_cache.get(cluster_spec)
    if cluster is None:
        if _host_pool:
            cluster = eco.start_deploy(
                hosts=_host_pool[: cluster_spec.count], wait=True
            )
        else:
            cluster = eco.start_deploy(
                count=cluster_spec.count,
                thunderbolt=cluster_spec.thunderbolt,
                chip=cluster_spec.chip,
                min_memory_gb=cluster_spec.min_memory_gb,
                wait=True,
            )
        _cluster_cache[cluster_spec] = cluster

    # Place an instance for this test if the test specified one
    instance_id = None
    if instance_spec is not None:
        client = cluster.make_client()
        instance_id = place_instance(
            client,
            instance_spec.model_id,
            sharding=instance_spec.sharding,
            comm=instance_spec.comm,
            min_nodes=instance_spec.min_nodes,
        )

    sess = Session(
        cluster=cluster,
        eco=eco,
        instance_spec=instance_spec,
        instance_id=instance_id,
    )

    yield sess

    # ---- Teardown ----

    # If the test left nodes disconnected, invalidate the cluster cache and
    # stop the cluster so the next test deploys fresh.
    if sess._stopped_hosts:
        _cluster_cache.pop(cluster_spec, None)
        with contextlib.suppress(Exception):
            eco.stop(sess.cluster.hosts)
        return

    # Otherwise, clean up any instances created during the test
    with contextlib.suppress(Exception):
        cleanup_all_instances(sess.client)


# ---------------------------------------------------------------------------
# Session-level teardown — stop all cached clusters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _teardown_clusters():
    yield
    for cluster in _cluster_cache.values():
        with contextlib.suppress(Exception):
            eco.stop(cluster.hosts)
    _cluster_cache.clear()


def pytest_runtest_makereport(item, call):
    """Attach cluster logs to the test report when a test fails."""
    if call.when != "call" or call.excinfo is None:
        return

    sess = item.funcargs.get("session")
    if sess is None:
        return
    try:
        logs = eco.logs(sess.cluster.hosts, lines=200)
        item.add_report_section("call", "Cluster Logs", json.dumps(logs, indent=2))
    except Exception:
        pass
