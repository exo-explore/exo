from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast

from pydantic import ConfigDict, Field, field_serializer, field_validator
from pydantic.alias_generators import to_camel

from exo.shared.topology import Topology, TopologySnapshot
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import (
    MemoryUsage,
    NodeIdentity,
    NodeNetworkInfo,
    NodePerformanceProfile,
    NodeThunderboltInfo,
    SystemPerformanceProfile,
)
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.pydantic_ext import CamelCaseModel


class State(CamelCaseModel):
    """Global system state.

    The :class:`Topology` instance is encoded/decoded via an immutable
    :class:`~shared.topology.TopologySnapshot` to ensure compatibility with
    standard JSON serialisation.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        extra="forbid",
        # I want to reenable this ASAP, but it's causing an issue with TaskStatus
        strict=True,
        arbitrary_types_allowed=True,
    )
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
    node_system: Mapping[NodeId, SystemPerformanceProfile] = {}
    node_network: Mapping[NodeId, NodeNetworkInfo] = {}
    node_thunderbolt: Mapping[NodeId, NodeThunderboltInfo] = {}

    @property
    def node_profiles(self) -> Mapping[NodeId, NodePerformanceProfile]:
        """Backwards-compatible property reconstructing NodePerformanceProfile from granular mappings."""
        all_node_ids: set[NodeId] = (
            set(self.node_identities.keys())
            | set(self.node_memory.keys())
            | set(self.node_system.keys())
            | set(self.node_network.keys())
            | set(self.node_thunderbolt.keys())
        )

        result: dict[NodeId, NodePerformanceProfile] = {}
        for node_id in all_node_ids:
            identity = self.node_identities.get(node_id, NodeIdentity())
            memory = self.node_memory.get(
                node_id,
                MemoryUsage.from_bytes(
                    ram_total=0, ram_available=0, swap_total=0, swap_available=0
                ),
            )
            system = self.node_system.get(node_id, SystemPerformanceProfile())
            network = self.node_network.get(node_id, NodeNetworkInfo())
            thunderbolt = self.node_thunderbolt.get(node_id, NodeThunderboltInfo())

            result[node_id] = NodePerformanceProfile(
                model_id=identity.model_id,
                chip_id=identity.chip_id,
                friendly_name=identity.friendly_name,
                memory=memory,
                network_interfaces=network.interfaces,
                tb_interfaces=thunderbolt.interfaces,
                system=system,
            )

        return result

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
            snapshot = TopologySnapshot(**cast(dict[str, Any], value))  # type: ignore[arg-type]
            return Topology.from_snapshot(snapshot)

        raise TypeError("Invalid representation for Topology field in State")
