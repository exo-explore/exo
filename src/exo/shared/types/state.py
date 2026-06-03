from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from exo_rs import LVAggregator
from pydantic import ConfigDict, Field, TypeAdapter, model_serializer
from pydantic.alias_generators import to_camel
from pydantic_core.core_schema import SerializerFunctionWrapHandler

from exo.shared.models import model_cards
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeDownloadProgress, NodeGatheredInfo
from exo.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    NodeIdentity,
    NodeNetworkInfo,
    NodeRdmaCtlStatus,
    NodeThunderboltInfo,
    SystemPerformanceProfile,
    ThunderboltBridgeStatus,
)
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.topology import (
    Connection,
    RDMAConnection,
    SocketConnection,
)
from exo.shared.types.worker.downloads import DownloadPending, DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    MacThunderboltConnections,
)
from exo.utils.pydantic_ext import FrozenModel

_DOWNLOAD_PROGRESS_ADAPTER = TypeAdapter[DownloadProgress](DownloadProgress)
_GATHERED_INFO_ADAPTER = TypeAdapter[GatheredInfo](GatheredInfo)


class State(FrozenModel):
    """Global system state.

    The :class:`Topology` instance is encoded/decoded via an immutable
    :class:`~shared.topology.TopologySnapshot` to ensure compatibility with
    standard JSON serialisation.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )
    instances: Mapping[InstanceId, Instance] = {}
    runners: Mapping[RunnerId, RunnerStatus] = {}
    downloads: Mapping[NodeId, Sequence[DownloadProgress]] = {}
    tasks: Mapping[TaskId, Task] = {}
    last_seen: Mapping[NodeId, datetime] = {}
    last_event_applied_idx: int = Field(default=-1, ge=-1)

    # Granular node state mappings (update independently at different frequencies)
    node_identities: Mapping[NodeId, NodeIdentity] = {}
    node_memory: Mapping[NodeId, MemoryUsage] = {}
    node_disk: Mapping[NodeId, DiskUsage] = {}
    node_system: Mapping[NodeId, SystemPerformanceProfile] = {}
    node_network: Mapping[NodeId, NodeNetworkInfo] = {}
    node_thunderbolt: Mapping[NodeId, NodeThunderboltInfo] = {}
    node_thunderbolt_bridge: Mapping[NodeId, ThunderboltBridgeStatus] = {}
    node_rdma_ctl: Mapping[NodeId, NodeRdmaCtlStatus] = {}
    node_backends: Mapping[NodeId, list[Backend]] = {}
    node_socket_connections: Mapping[
        NodeId, Mapping[NodeId, Sequence[SocketConnection]]
    ] = {}
    node_thunderbolt_connections: Mapping[NodeId, MacThunderboltConnections] = {}

    # Detected cycles where all nodes have Thunderbolt bridge enabled (>2 nodes)
    thunderbolt_bridge_cycles: Sequence[Sequence[NodeId]] = []

    prefill_server_ports: Mapping[RunnerId, int] = {}

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)  # pyright: ignore[reportAny]
        data["topology"] = {
            "nodes": list(self.node_identities.keys()),
            "connections": self.topology.map_connections(),
        }
        return data  # pyright: ignore[reportAny]

    @property
    def topology(self) -> Topology:
        topology = Topology()
        thunderbolt_by_uuid = {
            ident.domain_uuid: (node_id, ident.rdma_interface)
            for node_id, info in self.node_thunderbolt.items()
            for ident in info.interfaces
        }
        for node_id in self.node_identities:
            topology.add_node(node_id)

        for source, data in self.node_socket_connections.items():
            for sink, conns in data.items():
                for conn in conns:
                    topology.add_connection(
                        Connection(source=source, sink=sink, edge=conn)
                    )

        for source, connections in self.node_thunderbolt_connections.items():
            if not self.node_rdma_ctl.get(
                source, NodeRdmaCtlStatus(enabled=False)
            ).enabled:
                continue
            for connection in connections.conns:
                if (
                    source_iface := thunderbolt_by_uuid.get(connection.source_uuid)
                ) is None or (
                    sink_iface := thunderbolt_by_uuid.get(connection.sink_uuid)
                ) is None:
                    continue
                if not self.node_rdma_ctl.get(
                    sink_iface[0], NodeRdmaCtlStatus(enabled=False)
                ).enabled:
                    continue
                assert source_iface[0] == source, "registered invalid source uuid"
                topology.add_connection(
                    Connection(
                        source=source_iface[0],
                        sink=sink_iface[0],
                        edge=RDMAConnection(
                            source_rdma_iface=source_iface[1],
                            sink_rdma_iface=sink_iface[1],
                        ),
                    )
                )

        return topology

    def with_aggregator(self, aggregator: LVAggregator) -> "State":
        from exo.shared.apply import event_apply

        state = self.model_copy()
        values = aggregator.dump()
        node_ids = {NodeId(key.split("/")[0]) for key in values}
        cached_cards = model_cards.card_cache.list_cached()
        if cached_cards and node_ids:
            state = state.model_copy(
                update={
                    "downloads": {
                        node_id: [
                            DownloadPending(
                                node_id=node_id,
                                shard_metadata=PipelineShardMetadata(
                                    model_card=card,
                                    device_rank=0,
                                    world_size=1,
                                    start_layer=0,
                                    end_layer=card.n_layers,
                                    n_layers=card.n_layers,
                                ),
                                total=card.storage_size,
                            )
                            for card in cached_cards
                        ]
                        for node_id in node_ids
                    }
                }
            )

        for key, value in values.items():
            try:
                parts = key.split("/")
                if len(parts) >= 3 and parts[1] == "downloads":
                    progress = _DOWNLOAD_PROGRESS_ADAPTER.validate_json(value)
                    event = NodeDownloadProgress(download_progress=progress)
                else:
                    data = _GATHERED_INFO_ADAPTER.validate_json(value)
                    node_id = NodeId(parts[0])
                    event = NodeGatheredInfo(
                        node_id=node_id,
                        when=str(datetime.now(tz=timezone.utc)),
                        info=data,
                    )
                state = event_apply(event, state)
            except Exception as e:
                print(
                    f"\n{'=' * 10}key: {key} with exception {str(e)}\nvalue: {value}{'=' * 10}\n"
                )

        return state
