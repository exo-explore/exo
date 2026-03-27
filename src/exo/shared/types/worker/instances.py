from enum import Enum

from pydantic import model_validator

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.common import Host, Id, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"
    LlamaCppRpc = "LlamaCppRpc"


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


class LlamaCppRpcInstance(BaseInstance):
    """Instance backed by llama-cpp-python with optional RPC distribution.

    For a single-node setup, ``rpc_addresses`` is empty and the full model
    runs locally on rank 0.

    For a two-node setup:
    - Rank 1 spawns a ``llama-rpc-server`` on ``rpc_port``.
    - Rank 0 connects to each rank-1 host via ``rpc_addresses`` and splits
      GPU layers according to ``n_gpu_layers_per_runner``.
    """

    rpc_port: int
    # NodeId → "host:port" for each non-rank-0 node (empty for single-node)
    rpc_addresses: dict[NodeId, str]
    # RunnerId → number of GPU layers that runner should hold
    n_gpu_layers_per_runner: dict[RunnerId, int]


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance | LlamaCppRpcInstance


class BoundInstance(CamelCaseModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @property
    def is_image_model(self) -> bool:
        return (
            ModelTask.TextToImage in self.bound_shard.model_card.tasks
            or ModelTask.ImageToImage in self.bound_shard.model_card.tasks
        )

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        return self
