"""Two nodes in separate processes, with real discovery and zenoh: they connect only when they
run the same wire version and the same build.

Marked slow: each case waits out a fixed window, and discovery needs to send IPv6 multicast,
which macOS allows only to a process with Local Network access (for one started over ssh, or
from a terminal app that has it).
"""

import json
import socket
import subprocess
import sys
import uuid
from typing import cast

import pytest

from exo.routing.wire import WIRE_VERSION

# Matched nodes heard each other within 1.2 s on a laptop (a Hello goes out every second, then
# a zenoh session and the subscriptions form): 10 s is a wide margin, and is the window in
# which mismatched nodes must not connect.
_WINDOW_SECONDS = 10.0
# Two nodes on one machine share discovery's port. macOS hands a reply sent to that port to
# the socket bound first, so only the first node discovers the other, and a node dials only
# a peer whose id is smaller than its own: the node with the greater id starts first.
_FIRST_ID = "f" * 32
_SECOND_ID = "1" * 32


def _free_port(kind: socket.SocketKind) -> int:
    with socket.socket(socket.AF_INET6, kind) as sock:
        sock.bind(("::", 0))
        return cast(int, sock.getsockname()[1])


_BUILD = "test-build"


def _run_pair(
    first_version: int,
    second_version: int,
    first_build: str = _BUILD,
    second_build: str = _BUILD,
    package_versions: tuple[str, str] | None = None,
) -> list[dict[str, object]]:
    """Run two nodes until the window ends; return their reports. They share a fresh
    namespace, or with ``package_versions`` are source builds of those versions of exo on
    exo's default namespace (the discovery port keeps them apart from other nodes)."""
    namespace = f"exo-wire-gate-test-{uuid.uuid4().hex[:12]}"
    discovery_port = _free_port(socket.SOCK_DGRAM)

    def start(
        identity: str, version: int, build: str, package_version: str | None
    ) -> subprocess.Popen[str]:
        node_arguments = (
            ["--namespace", "default", "--package-version", package_version]
            if package_version is not None
            else ["--namespace", namespace, "--build-id", build]
        )
        peer = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "exo.routing.tests.wire_gate_peer",
                "--identity",
                identity,
                *node_arguments,
                "--wire-version",
                str(version),
                "--zenoh-port",
                str(_free_port(socket.SOCK_STREAM)),
                "--discovery-port",
                str(discovery_port),
                "--seconds",
                str(_WINDOW_SECONDS),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert peer.stdout is not None
        assert peer.stdout.readline().strip() == "ready"
        return peer

    first_package, second_package = package_versions or (None, None)
    peers = [
        start(_FIRST_ID, first_version, first_build, first_package),
        start(_SECOND_ID, second_version, second_build, second_package),
    ]
    reports: list[dict[str, object]] = []
    for peer in peers:
        output, _ = peer.communicate(timeout=_WINDOW_SECONDS + 30)
        assert peer.returncode == 0, output
        reports.append(
            cast(dict[str, object], json.loads(output.strip().splitlines()[-1]))
        )
    return reports


@pytest.mark.slow
def test_nodes_of_the_same_wire_version_and_build_connect() -> None:
    reports = _run_pair(WIRE_VERSION, WIRE_VERSION)
    for report in reports:
        assert report["heard_after"] is not None, reports
        assert report["other_versions"] == [], reports


def _heard(report: dict[str, object]) -> list[tuple[str, int | None]]:
    """The reasons and wire versions of the devices a node saw that can't join."""
    return [
        (cast(str, reason), cast(int | None, version))
        for _, reason, version in cast(list[list[object]], report["other_versions"])
    ]


@pytest.mark.slow
def test_nodes_on_different_wire_versions_never_connect() -> None:
    newer = WIRE_VERSION + 1
    older_report, newer_report = _run_pair(WIRE_VERSION, newer)
    assert older_report["heard_after"] is None, (older_report, newer_report)
    assert newer_report["heard_after"] is None, (older_report, newer_report)
    # Each says a device in its namespace runs another version, and which, once per device
    assert _heard(older_report) == [("wire_version", newer)]
    assert _heard(newer_report) == [("wire_version", WIRE_VERSION)]


@pytest.mark.slow
def test_nodes_of_different_builds_never_connect() -> None:
    first, second = _run_pair(
        WIRE_VERSION, WIRE_VERSION, "1.0.71 (aaaaaaa)", "1.0.72 (bbbbbbb)"
    )
    assert first["heard_after"] is None and second["heard_after"] is None
    # Each lists the other as another build, once
    assert _heard(first) == _heard(second) == [("build", None)]


@pytest.mark.slow
def test_source_builds_of_different_package_versions_name_each_other() -> None:
    # On exo's default name, the same for every version: they don't connect, and each says
    # the other runs another build instead of seeing another cluster
    first, second = _run_pair(
        WIRE_VERSION, WIRE_VERSION, package_versions=("0.3.70", "0.4.0")
    )
    assert first["heard_after"] is None and second["heard_after"] is None
    assert _heard(first) == _heard(second) == [("build", None)]
