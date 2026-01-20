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
    FLASH = "FLASH"


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


class MlxRingInstance(BaseInstance):
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


class MlxJacclInstance(BaseInstance):
    jaccl_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


class FLASHInstance(BaseInstance):
    """Instance for FLASH MPI simulation.

    Unlike MLX instances which do tensor parallelism, FLASH instances
    coordinate MPI processes across nodes. Each node runs one or more
    MPI ranks of the FLASH simulation.
    """

    hosts_by_node: dict[NodeId, list[Host]]
    flash_executable_path: str
    parameter_file_path: str
    working_directory: str
    ranks_per_node: int = 1
    total_ranks: int
    simulation_name: str
    coordinator_ip: str
    network_interface: str = "en0"  # Network interface for MPI (e.g., en0, eth0)


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance | FLASHInstance


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
