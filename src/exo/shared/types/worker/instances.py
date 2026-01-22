"""Instance types for exo.

Instances are registered dynamically via the instance_registry, allowing plugins
to add their own instance types without modifying this file.
"""

from enum import Enum
from typing import Any, cast

from pydantic import field_validator, model_validator

from exo.plugins.type_registry import instance_registry
from exo.shared.types.common import Host, Id, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"


class BaseInstance(TaggedModel):
    """Base class for all instance types."""

    instance_id: InstanceId
    shard_assignments: ShardAssignments

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


@instance_registry.register
class MlxRingInstance(BaseInstance):
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


@instance_registry.register
class MlxJacclInstance(BaseInstance):
    jaccl_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


# Union type for Pydantic validation - tries each type in order
# This is used by API endpoints (dashboard) which send flat format
Instance = MlxRingInstance | MlxJacclInstance


class BoundInstance(CamelCaseModel):
    """An instance bound to a specific runner on a specific node."""

    instance: BaseInstance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @field_validator("instance", mode="before")
    @classmethod
    def validate_instance(cls, v: Any) -> BaseInstance:  # noqa: ANN401  # pyright: ignore[reportAny]
        """Validate instance using registry to handle both tagged and flat formats."""
        return cast(BaseInstance, instance_registry.deserialize(v))  # pyright: ignore[reportAny]

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        return self
