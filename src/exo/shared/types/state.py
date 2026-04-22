from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import Field, field_serializer, field_validator

from exo.shared.topology import Topology, TopologySnapshot
from exo.shared.types.common import NodeId, SessionId
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
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.pydantic_ext import FrozenModel


class BaseState(ABC):
    @abstractmethod
    def last_event_idx(self) -> int: ...


class State(BaseState, FrozenModel, arbitrary_types_allowed=True):
    """Global system state.

    The :class:`Topology` instance is encoded/decoded via an immutable
    :class:`~shared.topology.TopologySnapshot` to ensure compatibility with
    standard JSON serialisation.
    """

    instances: Mapping[InstanceId, Instance] = {}
    runners: Mapping[RunnerId, RunnerStatus] = {}
    downloads: Mapping[NodeId, Sequence[DownloadProgress]] = {}
    tasks: Mapping[TaskId, Task] = {}
    last_seen: Mapping[NodeId, datetime] = {}
    topology: Topology = Field(default_factory=Topology)
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

    # Detected cycles where all nodes have Thunderbolt bridge enabled (>2 nodes)
    thunderbolt_bridge_cycles: Sequence[Sequence[NodeId]] = []

    def last_event_idx(self) -> int:
        return self.last_event_applied_idx

    @field_serializer("topology", mode="plain")
    def _encode_topology(self, value: Topology) -> TopologySnapshot:
        return value.to_snapshot()

    @field_validator("topology", mode="before")
    @classmethod
    def _deserialize_topology(cls, value: object) -> Topology:  # noqa: D401 – Pydantic validator signature
        """Convert an incoming *value* into a :class:`Topology` instance.

        Accepts either an already constructed :class:`Topology` or a mapping
        representing :class:`~shared.topology.TopologySnapshot`.
        """

        if isinstance(value, Topology):
            return value

        if isinstance(value, Mapping):  # likely a snapshot-dict coming from JSON
            snapshot = TopologySnapshot.model_validate(value)
            return Topology.from_snapshot(snapshot)

        raise TypeError("Invalid representation for Topology field in State")


class ForwarderState(FrozenModel):
    state: State
    session_id: SessionId
