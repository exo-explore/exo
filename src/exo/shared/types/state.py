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
    NodeThunderboltInfo,
    SystemPerformanceProfile,
)
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import BaseInstance, InstanceId
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
    instances: Mapping[InstanceId, BaseInstance] = {}
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

    @field_serializer("instances", mode="plain")
    def _encode_instances(
        self, value: Mapping[InstanceId, BaseInstance]
    ) -> dict[str, Any]:
        """Serialize instances with full subclass fields."""
        return {
            str(k): v.model_dump(by_alias=True, serialize_as_any=True)
            for k, v in value.items()
        }

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
