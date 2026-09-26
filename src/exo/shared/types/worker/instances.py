from enum import Enum

from pydantic import model_validator

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import Host, Id, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"
    Tinygrad = "Tinygrad"


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


class TinygradInstance(BaseInstance):
    """Placement executed by the tinygrad adapter.

    ``device_backend_by_node`` records the backend selected for each node.
    The runner turns that backend into a tinygrad device name at startup.
    """

    device_backend_by_node: dict[NodeId, Backend]

    def backend_for_node(self, node_id: NodeId) -> Backend:
        """Return the tinygrad backend selected for ``node_id``.

        Raises:
            ValueError: The runner entrypoint handles this by publishing
                ``RunnerTerminationError`` when the placement omitted the
                node that is starting the runner.
        """
        try:
            return self.device_backend_by_node[node_id]
        except KeyError as error:
            raise ValueError(
                f"Tinygrad instance {self.instance_id} has no backend for node {node_id}"
            ) from error


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance | TinygradInstance


class BoundInstance(FrozenModel):
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
