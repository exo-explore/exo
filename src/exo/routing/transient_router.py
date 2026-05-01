from dataclasses import dataclass, field

from anyio import BrokenResourceError, ClosedResourceError
from loguru import logger

from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    GlobalForwarderTransientEvent,
    TransientEvent,
)
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.task_group import TaskGroup


@dataclass
class TransientRouter:
    """Routes TransientEvents over the cluster's TRANSIENT_EVENTS topic.

    Transients are per-request streaming/notification signals (token chunks,
    image-upload chunks, trace data) that intentionally bypass the durable
    event log: they're not ordered, not persisted, not replayed. A node that
    joins via snapshot will simply not see transients that were emitted before
    it joined, which is the correct behaviour — replaying old token chunks to
    a closed HTTP stream would be wrong.

    Producers call `sender()` to get a `Sender[TransientEvent]`; values sent
    on it are wrapped in a `GlobalForwarderTransientEvent` and published on
    the network. Consumers call `receiver()` to get a `Receiver[TransientEvent]`
    that yields events from this session only.
    """

    node_id: NodeId
    session_id: SessionId
    external_outbound: Sender[GlobalForwarderTransientEvent]
    external_inbound: Receiver[GlobalForwarderTransientEvent]
    _outbound: list[Sender[TransientEvent]] = field(init=False, default_factory=list)
    _inbound: list[Receiver[TransientEvent]] = field(init=False, default_factory=list)
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    def sender(self) -> Sender[TransientEvent]:
        send, recv = channel[TransientEvent]()
        if self._tg.is_running():
            self._tg.start_soon(self._publish, recv)
        else:
            self._inbound.append(recv)
        return send

    def receiver(self) -> Receiver[TransientEvent]:
        assert not self._tg.is_running(), (
            "TransientRouter receivers must be registered before run()"
        )
        send, recv = channel[TransientEvent]()
        self._outbound.append(send)
        return recv

    def shutdown(self) -> None:
        self._tg.cancel_tasks()

    async def run(self) -> None:
        try:
            async with self._tg as tg:
                for recv in self._inbound:
                    tg.start_soon(self._publish, recv)
                tg.start_soon(self._dispatch_inbound)
        finally:
            self.external_outbound.close()
            for send in self._outbound:
                send.close()

    async def _publish(self, recv: Receiver[TransientEvent]) -> None:
        with recv as events:
            async for event in events:
                await self.external_outbound.send(
                    GlobalForwarderTransientEvent(
                        origin=self.node_id,
                        session=self.session_id,
                        event=event,
                    )
                )

    async def _dispatch_inbound(self) -> None:
        with self.external_inbound as wrapped_events:
            async for wrapped in wrapped_events:
                if wrapped.session != self.session_id:
                    continue
                stale: set[int] = set()
                for i, send in enumerate(self._outbound):
                    try:
                        await send.send(wrapped.event)
                    except (ClosedResourceError, BrokenResourceError):
                        stale.add(i)
                if stale:
                    for i in sorted(stale, reverse=True):
                        self._outbound.pop(i)
                    logger.debug(
                        f"TransientRouter dropped {len(stale)} closed receivers"
                    )
