"""GET /membership shows this node's own view of the cluster: each node's state and why, and how
much of the cluster it reaches."""

from typing import cast

import exo_rs
from exo_rs import LossReason, MembershipSnapshot, Quorum
from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.routing.membership import Member, Membership
from exo.routing.tests.fake_networking import FakeNetworking
from exo.shared.types.common import NodeId
from exo.shared.types.events import IndexedEvent, StateSnapshot
from exo.utils.channels import channel

SINCE = 1_800_000_000.0


def snapshot() -> MembershipSnapshot:
    return MembershipSnapshot(
        [
            exo_rs.Member("a1", exo_rs.MemberState.Alive, None, SINCE),
            exo_rs.Member("b2", exo_rs.MemberState.Alive, None, SINCE),
            exo_rs.Member(
                "c3", exo_rs.MemberState.Suspect, LossReason.Disconnected, SINCE + 2.5
            ),
        ],
        Quorum.Majority,
        2,
        3,
    )


def served_membership() -> object:
    _, receiver = channel[Member]()
    _, cluster_events = channel[IndexedEvent | StateSnapshot]()
    api = object.__new__(API)
    api.membership = Membership(
        NodeId("a1"),
        member_receiver=receiver,
        networking=FakeNetworking(snapshot),
        cluster_events=cluster_events,
    )
    app = FastAPI()
    app.get("/membership")(api.get_membership)
    return cast(object, TestClient(app).get("/membership").json())


def test_membership_shows_each_node_and_the_quorum_read() -> None:
    assert served_membership() == {
        "nodeId": "a1",
        "selfState": "Alive",
        "quorum": "Majority",
        "reachable": 2,
        "lastKnown": 3,
        "members": [
            {
                "nodeId": "a1",
                "state": "Alive",
                "reason": None,
                "since": "2027-01-15T08:00:00Z",
                "counted": True,
            },
            {
                "nodeId": "b2",
                "state": "Alive",
                "reason": None,
                "since": "2027-01-15T08:00:00Z",
                "counted": True,
            },
            {
                "nodeId": "c3",
                "state": "Suspect",
                "reason": "Disconnected",
                "since": "2027-01-15T08:00:02.500000Z",
                "counted": True,
            },
        ],
    }
