"""Which nodes serving treats as reachable: the membership's view, except for the nodes it loses
while this node is itself Resuming (it was frozen or asleep), which are judged once it is back."""

from typing import final

import anyio

from exo.routing.membership import MembershipChange, MemberState
from exo.shared.serving import ServingReachability
from exo.shared.types.common import NodeId
from exo.utils.channels import Receiver, Sender, channel

ME, PEER, OTHER = NodeId("me"), NodeId("peer"), NodeId("other")


@final
class FakeMembership:
    def __init__(self, alive: set[NodeId]) -> None:
        self.alive = set(alive)
        self._senders: list[Sender[MembershipChange]] = []

    def is_reachable(self, node: NodeId) -> bool:
        return node in self.alive

    def changes(self) -> Receiver[MembershipChange]:
        send, receive = channel[MembershipChange]()
        self._senders.append(send)
        return receive

    async def change(
        self, node: NodeId, previous: MemberState, current: MemberState
    ) -> None:
        if current in ("Alive", "Resuming"):
            self.alive.add(node)
        else:
            self.alive.discard(node)
        for sender in self._senders:
            await sender.send(
                MembershipChange(
                    node_id=node,
                    previous=previous,
                    current=current,
                    reason=None if current in ("Alive", "Resuming") else "Disconnected",
                )
            )
        await anyio.sleep(0.01)


async def test_a_peer_lost_while_this_node_runs_is_reported_at_once() -> None:
    membership = FakeMembership({ME, PEER})
    reachability = ServingReachability(ME, membership)
    async with anyio.create_task_group() as tg:
        tg.start_soon(reachability.run)
        await anyio.sleep(0.01)
        lost = reachability.watch()
        await membership.change(PEER, "Alive", "Suspect")
        with anyio.fail_after(1):
            assert await lost.receive() == PEER
        assert not reachability.is_reachable(PEER)
        tg.cancel_scope.cancel()


async def test_peers_lost_while_this_node_resumes_are_judged_once_it_is_back() -> None:
    """This node was frozen past the lease: as it resumes, its membership marks both peers
    Suspect, as its own links lapsed. Neither is reported then. Once it is back, the peer that
    is Alive again isn't reported at all, and the one still unreachable is."""
    membership = FakeMembership({ME, PEER, OTHER})
    reachability = ServingReachability(ME, membership)
    async with anyio.create_task_group() as tg:
        tg.start_soon(reachability.run)
        await anyio.sleep(0.01)
        lost = reachability.watch()
        await membership.change(ME, "Alive", "Resuming")
        await membership.change(PEER, "Alive", "Suspect")
        await membership.change(OTHER, "Alive", "Suspect")
        assert lost.collect() == []

        await membership.change(PEER, "Suspect", "Alive")
        await membership.change(ME, "Resuming", "Alive")
        with anyio.fail_after(1):
            assert await lost.receive() == OTHER
        assert lost.collect() == []
        tg.cancel_scope.cancel()
