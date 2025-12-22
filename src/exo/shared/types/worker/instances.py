from enum import Enum

from pydantic import model_validator

from exo.shared.types.common import Host, Id, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"
    LlamaCpp = "LlamaCpp"


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


class MlxRingInstance(BaseInstance):
    hosts: list[Host]


class MlxJacclInstance(BaseInstance):
    ibv_devices: list[list[str | None]]
    ibv_coordinators: dict[NodeId, str]


class LlamaCppInstance(BaseInstance):
    """
    Instance type for llama.cpp-based inference.
    Used for cross-platform inference on Android, Linux, and other non-Apple platforms.

    For multi-node distributed inference using llama.cpp RPC backend:
    - Worker nodes (device_rank > 0) run rpc-server on their assigned port
    - Master node (device_rank == 0) connects to workers via --rpc flag
    - Layers are distributed via tensor_split based on memory ratios
    """

    hosts: list[Host]
    rpc_ports: dict[NodeId, int] = {}
    tensor_split: list[float] = []

    @property
    def is_distributed(self) -> bool:
        """Check if this instance uses distributed inference across multiple nodes."""
        return len(self.rpc_ports) > 0 and len(self.tensor_split) > 1


Instance = MlxRingInstance | MlxJacclInstance | LlamaCppInstance


class BoundInstance(CamelCaseModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

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
