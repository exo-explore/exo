import asyncio
import logging

from exo_pyo3_bindings import ConnectionUpdate, DiscoveryService, Keypair

from shared.db import AsyncSQLiteEventStorage
from shared.types.common import NodeId
from shared.types.events import TopologyEdgeCreated, TopologyEdgeDeleted
from shared.types.multiaddr import Multiaddr
from shared.types.topology import Connection


class DiscoverySupervisor:
    def __init__(self, node_id_keypair: Keypair, node_id: NodeId, global_events: AsyncSQLiteEventStorage,
                 logger: logging.Logger):
        self.global_events = global_events
        self.logger = logger
        self.node_id = node_id

        # configure callbacks
        self.discovery_service = DiscoveryService(node_id_keypair)
        self._add_connected_callback()
        self._add_disconnected_callback()

    def _add_connected_callback(self):
        stream_get, stream_put = _make_iter()
        self.discovery_service.add_connected_callback(stream_put)

        async def run():
            async for c in stream_get:
                await self._connected_callback(c)

        return asyncio.create_task(run())

    def _add_disconnected_callback(self):
        stream_get, stream_put = _make_iter()

        async def run():
            async for c in stream_get:
                await self._disconnected_callback(c)

        self.discovery_service.add_disconnected_callback(stream_put)
        return asyncio.create_task(run())

    async def _connected_callback(self, e: ConnectionUpdate) -> None:
        local_node_id = self.node_id
        send_back_node_id = NodeId(e.peer_id.to_base58())
        local_multiaddr = Multiaddr(address=str(e.local_addr))
        send_back_multiaddr = Multiaddr(address=str(e.send_back_addr))
        connection_profile = None
        
        if send_back_multiaddr.ipv4_address == local_multiaddr.ipv4_address:
            return
        
        topology_edge_created = TopologyEdgeCreated(edge=Connection(
            local_node_id=local_node_id,
            send_back_node_id=send_back_node_id,
            local_multiaddr=local_multiaddr,
            send_back_multiaddr=send_back_multiaddr,
            connection_profile=connection_profile
        ))
        self.logger.info(
            msg=f"CONNECTED CALLBACK: {local_node_id} -> {send_back_node_id}, {local_multiaddr} -> {send_back_multiaddr}")
        await self.global_events.append_events(
            [topology_edge_created],
            self.node_id
        )

    async def _disconnected_callback(self, e: ConnectionUpdate) -> None:
        local_node_id = self.node_id
        send_back_node_id = NodeId(e.peer_id.to_base58())
        local_multiaddr = Multiaddr(address=str(e.local_addr))
        send_back_multiaddr = Multiaddr(address=str(e.send_back_addr))
        connection_profile = None

        topology_edge_created = TopologyEdgeDeleted(edge=Connection(
            local_node_id=local_node_id,
            send_back_node_id=send_back_node_id,
            local_multiaddr=local_multiaddr,
            send_back_multiaddr=send_back_multiaddr,
            connection_profile=connection_profile
        ))
        self.logger.error(
            msg=f"DISCONNECTED CALLBACK: {local_node_id} -> {send_back_node_id}, {local_multiaddr} -> {send_back_multiaddr}")
        await self.global_events.append_events(
            [topology_edge_created],
            self.node_id
        )


def _make_iter():  # TODO: generalize to generic utility
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[ConnectionUpdate] = asyncio.Queue()

    def put(c: ConnectionUpdate) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, c)

    async def get():
        while True:
            yield await queue.get()

    return get(), put

# class MyClass: # TODO: figure out how to make pydantic integrate with Multiaddr
#     def __init__(self, data: str):
#         self.data = data
#
#     @staticmethod
#     def from_str(s: str, _i: ValidationInfo) -> 'MyClass':
#         return MyClass(s)
#
#     def __str__(self):
#         return self.data
#
#     @classmethod
#     def __get_pydantic_core_schema__(
#             cls, source_type: type[any], handler: GetCoreSchemaHandler
#     ) -> CoreSchema:
#         return core_schema.with_info_after_validator_function(
#             function=MyClass.from_str,
#             schema=core_schema.bytes_schema(),
#             serialization=core_schema.to_string_ser_schema()
#         )
#
#
# # Use directly in a model (no Annotated needed)
# class ExampleModel(BaseModel):
#     field: MyClass
#
#
# m = ExampleModel(field=MyClass("foo"))
# d = m.model_dump()
# djs = m.model_dump_json()
#
# print(d)
# print(djs)
