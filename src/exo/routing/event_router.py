from dataclasses import dataclass, field
from random import random
import anyio
from anyio import ClosedResourceError, BrokenResourceError
from exo.utils.channels import Sender, Receiver
from exo.utils.lazy_task_group import LazyTaskGroup
from exo.utils.event_buffer import OrderedBuffer
from exo.shared.types.common import SessionId, SystemId
from exo.shared.types.events import LocalForwarderEvent, GlobalForwarderEvent, Event, EventId

@dataclass
class EventRouter:
    _tg: LazyTaskGroup = field(init=False, default_factory=LazyTaskGroup)
    internal_outbound: dict[SystemId, Sender[Event]] = field(init=False,default_factory=dict)
    external_inbound: Receiver[GlobalForwarderEvent]
    external_outbound: Sender[LocalForwarderEvent]
    event_buffer: OrderedBuffer[Event]
    session_id: SessionId
    out_for_delivery: dict[EventId, tuple[float, LocalForwarderEvent]] = field(init=False, default_factory=dict)


    async def run(self):
        try:
            async with self._tg as tg:
                tg.start_soon(self._run_ext_in)
                tg.start_soon(self._simple_retry)
        finally:
            self.external_outbound.close()
            for send in self.internal_outbound.values():
                send.close()

    # can make this better in future
    async def _simple_retry(self):
        while True:
            await anyio.sleep(1 + random())
            # list here is a shallow clone for shared mutation
            for id, (time, event) in list(self.out_for_delivery.items()):
                if anyio.current_time() > time + 5:
                    self.out_for_delivery[id] = (anyio.current_time(), event)
                    await self.external_outbound.send(event)


    def ingest(self, system_id: SystemId, recv: Receiver[Event]):
        self._tg.start_soon(self._ingest, system_id, recv)

    async def _ingest(self, system_id: SystemId, recv: Receiver[Event]):
        idx = 0
        with recv as events:
            async for event in events:
                f_ev = LocalForwarderEvent(origin_idx = idx, origin=system_id, session=self.session_id, event=event)
                idx += 1
                await self.external_outbound.send(f_ev)
                self.out_for_delivery[event.event_id] = (anyio.current_time(), f_ev)
        

        
    async def _run_ext_in(self):
        with self.external_inbound as events:
            async for event in events:
                if event.session != self.session_id:
                    continue
                if event.origin != self.session_id.master_node_id:
                    continue

                self.event_buffer.ingest(event.origin_idx, event.event)
                event_id = event.event.event_id
                if event_id in self.out_for_delivery:
                    self.out_for_delivery.pop(event_id)

                for event in self.event_buffer.drain():
                    to_clear = set[SystemId]()
                    for s_id, sender in self.internal_outbound.items():
                        try:
                            await sender.send(event)
                        except (ClosedResourceError, BrokenResourceError):
                            to_clear.add(s_id)
                    for s_id in to_clear:
                        self.internal_outbound.pop(s_id)


