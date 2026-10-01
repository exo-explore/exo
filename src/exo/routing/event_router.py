from dataclasses import dataclass, field
from random import random

import anyio
from anyio import BrokenResourceError, ClosedResourceError
from anyio.abc import CancelScope
from loguru import logger

from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import SessionId, SystemId
from exo.shared.types.events import (
    Event,
    EventId,
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    StateSnapshot,
)
from exo.utils import channels
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.event_buffer import OrderedBuffer
from exo.utils.task_group import TaskGroup


class EventRouterClosedResourceError(ClosedResourceError):
    pass


class EventRouterBrokenResourceError(BrokenResourceError):
    pass


# Event Router is created and destroyed before consumers of its channels are,
# hence its nice to have tagged errors for event-router channels being closed
#
# so consumers can catch specifically these errors, rather than the generic ones
_ERROR_CFG = channels.ErrorOverride(
    closed_resource_error=EventRouterClosedResourceError,
    broken_resource_error=EventRouterBrokenResourceError,
)


# What the worker and the API read: indexed events, plus a state snapshot whenever the node
# catches up from one. A receiver of events alone (with no snapshots) is accepted too.
type NodeEventReceiver = Receiver[IndexedEvent | StateSnapshot] | Receiver[IndexedEvent]


@dataclass
class EventRouter:
    session_id: SessionId
    command_sender: Sender[ForwarderCommand]
    external_inbound: Receiver[GlobalForwarderEvent]
    external_outbound: Sender[LocalForwarderEvent]
    snapshot_inbound: Receiver[StateSnapshot]
    _system_id: SystemId = field(init=False, default_factory=SystemId)
    internal_outbound: list[Sender[IndexedEvent | StateSnapshot]] = field(
        init=False, default_factory=list
    )
    event_buffer: OrderedBuffer[Event] = field(
        init=False, default_factory=OrderedBuffer
    )
    out_for_delivery: dict[EventId, tuple[float, LocalForwarderEvent]] = field(
        init=False, default_factory=dict
    )
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    _nack_cancel_scope: CancelScope | None = field(init=False, default=None)
    # Set by switch_session: the inbound buffer starts again for the new session
    _session_changed: bool = field(init=False, default=False)
    # After switch_session, until the new master's snapshot has been delivered
    _awaiting_snapshot: bool = field(init=False, default=False)
    _nack_attempts: int = field(init=False, default=0)
    _nack_base_seconds: float = field(init=False, default=0.5)
    _nack_cap_seconds: float = field(init=False, default=10.0)

    async def run(self):
        # Events and snapshots are handled by one loop so consumers always get a
        # snapshot before the events that follow it.
        inbound_send, inbound_recv = channel[GlobalForwarderEvent | StateSnapshot]()
        try:
            async with self._tg as tg:
                tg.start_soon(_forward, self.external_inbound, inbound_send.clone())
                tg.start_soon(_forward, self.snapshot_inbound, inbound_send)
                tg.start_soon(self._run_ext_in, inbound_recv)
                tg.start_soon(self._simple_retry)
        finally:
            self.external_outbound.close()
            for send in self.internal_outbound:
                send.close()

    # can make this better in future
    async def _simple_retry(self):
        while True:
            await anyio.sleep(1 + random())
            # list here is a shallow clone for shared mutation
            for e_id, (time, event) in list(self.out_for_delivery.items()):
                if anyio.current_time() > time + 5:
                    self.out_for_delivery[e_id] = (anyio.current_time(), event)
                    await self.external_outbound.send(event)

    def sender(self) -> Sender[Event]:
        send, recv = channel[Event](error_override_config=_ERROR_CFG)
        if self._tg.is_running():
            self._tg.start_soon(self._ingest, SystemId(), recv)
        else:
            self._tg.queue(self._ingest, SystemId(), recv)
        return send

    def receiver(self) -> Receiver[IndexedEvent | StateSnapshot]:
        assert not self._tg.is_running()
        send, recv = channel[IndexedEvent | StateSnapshot](
            error_override_config=_ERROR_CFG
        )
        self.internal_outbound.append(send)
        return recv

    def shutdown(self) -> None:
        self._tg.cancel_tasks()

    def switch_session(self, session_id: SessionId) -> None:
        """Follow a new master without tearing down this router or its channels.

        The new master numbers its events on from its own state, so this node catches up
        from a snapshot of that state; events from the old session are ignored from now
        on, and events not yet acknowledged by the old master are dropped (a new master's
        state is repaired by what nodes report next, not by replaying them).
        """
        self.session_id = session_id
        self.out_for_delivery.clear()
        self._session_changed = True
        # Consumers hold the old session's state, so they must get the new master's state
        # before any of its events: nothing is delivered until its snapshot arrives
        self._awaiting_snapshot = True
        self._caught_up()
        if self._tg.is_running():
            self._tg.start_soon(self._request_snapshot, session_id)

    async def _ingest(self, system_id: SystemId, recv: Receiver[Event]):
        idx = 0
        session = self.session_id
        with recv as events:
            async for event in events:
                if self.session_id != session:
                    # A master orders each origin's events from 0. The events get a new
                    # origin too: a master this node followed before (and may follow
                    # again) still expects the old one's next index.
                    session = self.session_id
                    system_id = SystemId()
                    idx = 0
                f_ev = LocalForwarderEvent(
                    origin_idx=idx,
                    origin=system_id,
                    session=self.session_id,
                    event=event,
                )
                idx += 1
                await self.external_outbound.send(f_ev)
                self.out_for_delivery[event.event_id] = (anyio.current_time(), f_ev)

    async def _run_ext_in(
        self, inbound: Receiver[GlobalForwarderEvent | StateSnapshot]
    ):
        buf = OrderedBuffer[Event]()
        with inbound as messages:
            async for message in messages:
                if self._session_changed:
                    self._session_changed = False
                    buf = OrderedBuffer[Event]()
                if message.session != self.session_id:
                    continue
                match message:
                    case StateSnapshot():
                        if message.requester != self._system_id:
                            continue
                        snapshot_idx = message.state.last_event_applied_idx
                        if (
                            snapshot_idx < buf.next_idx_to_release
                            and not self._awaiting_snapshot
                        ):
                            # We already have everything it covers
                            continue
                        self._awaiting_snapshot = False
                        logger.info(
                            f"Catching up from a state snapshot at event {snapshot_idx}"
                        )
                        buf.skip_to(snapshot_idx + 1)
                        await self._deliver(message)
                        drained = buf.drain_indexed()
                        self._caught_up()
                    case GlobalForwarderEvent():
                        if message.origin != self.session_id.master_node_id:
                            continue

                        buf.ingest(message.origin_idx, message.event)
                        event_id = message.event.event_id
                        if event_id in self.out_for_delivery:
                            self.out_for_delivery.pop(event_id)
                        if self._awaiting_snapshot:
                            # Held until the new master's snapshot, which may cover it
                            continue

                        drained = buf.drain_indexed()
                        if drained:
                            self._caught_up()

                        if not drained and (
                            self._nack_cancel_scope is None
                            or self._nack_cancel_scope.cancel_called
                        ):
                            # Request the next index.
                            self._tg.start_soon(
                                self._nack_request, buf.next_idx_to_release
                            )
                            continue

                for idx, event in drained:
                    await self._deliver(IndexedEvent(idx=idx, event=event))

    def _caught_up(self) -> None:
        self._nack_attempts = 0
        if self._nack_cancel_scope:
            self._nack_cancel_scope.cancel()

    async def _deliver(self, item: IndexedEvent | StateSnapshot) -> None:
        to_clear = set[int]()
        for i, sender in enumerate(self.internal_outbound):
            try:
                await sender.send(item)
            except (ClosedResourceError, BrokenResourceError):
                to_clear.add(i)
        for i in sorted(to_clear, reverse=True):
            self.internal_outbound.pop(i)

    async def _request_snapshot(self, session_id: SessionId) -> None:
        """Ask the new master for its state until it arrives (or the master changes again)."""
        delay = self._nack_base_seconds
        while self._awaiting_snapshot and self.session_id == session_id:
            logger.info("Asking the new master for a snapshot of its state")
            await self.command_sender.send(
                ForwarderCommand(
                    origin=self._system_id,
                    command=RequestEventLog(since_idx=0, snapshot=True),
                )
            )
            await anyio.sleep(delay)
            # The new master is often still starting: ask again soon
            delay = min(2.0, delay * 2)

    async def _nack_request(self, since_idx: int) -> None:
        # We request all events after (and including) the missing index.
        # This function is started whenever we receive an event that is out of sequence.
        # It is cancelled as soon as we receiver an event that is in sequence.

        if since_idx < 0:
            logger.warning(f"Negative value encountered for nack request {since_idx=}")
            since_idx = 0

        with CancelScope() as scope:
            self._nack_cancel_scope = scope
            delay: float = self._nack_base_seconds * (2.0**self._nack_attempts)
            delay = min(self._nack_cap_seconds, delay)
            self._nack_attempts += 1
            try:
                await anyio.sleep(delay)
                logger.info(
                    f"Nack attempt {self._nack_attempts}: Requesting Event Log from {since_idx}"
                )
                await self.command_sender.send(
                    ForwarderCommand(
                        origin=self._system_id,
                        command=RequestEventLog(since_idx=since_idx),
                    )
                )
            finally:
                if self._nack_cancel_scope is scope:
                    self._nack_cancel_scope = None


async def _forward[T: GlobalForwarderEvent | StateSnapshot](
    source: Receiver[T], sink: Sender[GlobalForwarderEvent | StateSnapshot]
) -> None:
    with source as items, sink:
        async for item in items:
            await sink.send(item)
