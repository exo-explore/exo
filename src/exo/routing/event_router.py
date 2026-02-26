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
)
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.event_buffer import OrderedBuffer
from exo.utils.task_group import TaskGroup


@dataclass
class EventRouter:
    session_id: SessionId
    command_sender: Sender[ForwarderCommand]
    external_inbound: Receiver[GlobalForwarderEvent]
    external_outbound: Sender[LocalForwarderEvent]
    _system_id: SystemId = field(init=False, default_factory=SystemId)
    internal_outbound: list[Sender[IndexedEvent]] = field(
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
    _nack_attempts: int = field(init=False, default=0)
    _nack_base_seconds: float = field(init=False, default=0.5)
    _nack_cap_seconds: float = field(init=False, default=10.0)

    async def run(self):
        try:
            async with self._tg as tg:
                tg.start_soon(self._run_ext_in)
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
        send, recv = channel[Event]()
        if self._tg.is_running():
            self._tg.start_soon(self._ingest, SystemId(), recv)
        else:
            self._tg.queue(self._ingest, SystemId(), recv)
        return send

    def receiver(self) -> Receiver[IndexedEvent]:
        send, recv = channel[IndexedEvent]()
        self.internal_outbound.append(send)
        return recv

    def shutdown(self) -> None:
        self._tg.cancel_tasks()

    async def _ingest(self, system_id: SystemId, recv: Receiver[Event]):
        idx = 0
        with recv as events:
            async for event in events:
                f_ev = LocalForwarderEvent(
                    origin_idx=idx,
                    origin=system_id,
                    session=self.session_id,
                    event=event,
                )
                idx += 1
                await self.external_outbound.send(f_ev)
                self.out_for_delivery[event.event_id] = (anyio.current_time(), f_ev)

    async def _run_ext_in(self):
        buf = OrderedBuffer[Event]()
        with self.external_inbound as events:
            async for event in events:
                if event.session != self.session_id:
                    continue
                if event.origin != self.session_id.master_node_id:
                    continue

                buf.ingest(event.origin_idx, event.event)
                event_id = event.event.event_id
                if event_id in self.out_for_delivery:
                    self.out_for_delivery.pop(event_id)

                drained = buf.drain_indexed()
                if drained:
                    self._nack_attempts = 0
                    if self._nack_cancel_scope:
                        self._nack_cancel_scope.cancel()

                if not drained and (
                    self._nack_cancel_scope is None
                    or self._nack_cancel_scope.cancel_called
                ):
                    # Request the next index.
                    self._tg.start_soon(self._nack_request, buf.next_idx_to_release)
                    continue

                for idx, event in drained:
                    to_clear = set[int]()
                    for i, sender in enumerate(self.internal_outbound):
                        try:
                            await sender.send(IndexedEvent(idx=idx, event=event))
                        except (ClosedResourceError, BrokenResourceError):
                            to_clear.add(i)
                    for i in sorted(to_clear, reverse=True):
                        self.internal_outbound.pop(i)

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
