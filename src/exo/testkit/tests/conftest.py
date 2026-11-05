import contextlib
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Protocol, cast

import pytest

from exo.testkit.cluster import Cluster

_CLUSTERS = pytest.StashKey[list[Cluster]]()
_LOG_TAIL = 4000


class MakeCluster(Protocol):
    def __call__(
        self, size: int, *, probe_own_port: bool = False, require_edges: bool = True
    ) -> Cluster: ...


@pytest.fixture
def make_cluster(
    tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[MakeCluster]:
    """Start a cluster of N stub nodes for this test, formed and ready; every cluster is
    stopped afterwards, and a failing test's report gets their timelines and logs. The test
    fails at teardown if the harness had to force something no scenario asked for (a node
    that didn't exit on SIGINT, a runner that outlived its node; see Cluster.fallbacks),
    unless it set `cluster.expect_fallbacks`."""
    clusters: list[Cluster] = []
    cast(pytest.Item, request.node).stash[_CLUSTERS] = clusters

    def make(
        size: int, *, probe_own_port: bool = False, require_edges: bool = True
    ) -> Cluster:
        cluster = Cluster(
            size, tmp_path / f"cluster{len(clusters)}", probe_own_port=probe_own_port
        )
        clusters.append(cluster)
        cluster.start(require_edges=require_edges)
        return cluster

    yield make
    problems: list[str] = []
    for cluster in clusters:
        # Close every cluster, even after one fails to close, then report them all
        try:
            cluster.close()
        except Exception as error:
            problems.append(f"{cluster.root}: close failed: {error!r}")
        if cluster.fallbacks and not cluster.expect_fallbacks:
            problems.extend(
                f"{cluster.root}: {fallback}" for fallback in cluster.fallbacks
            )
    if problems:
        pytest.fail("teardown: " + "; ".join(problems))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """On a failure, attach each cluster's timeline and the end of every node's log (on a
    teardown failure, just the timelines, which hold the fallbacks)."""
    report = yield
    if report.when in ("call", "teardown") and report.failed:
        none: list[Cluster] = []
        for cluster in item.stash.get(_CLUSTERS, none):
            with contextlib.suppress(OSError):
                report.sections.append(
                    (f"timeline {cluster.root}", cluster.timeline_path.read_text())
                )
            if report.when == "call":
                for name, log in cluster.logs():
                    report.sections.append((f"{name} log (tail)", log[-_LOG_TAIL:]))
    return report
