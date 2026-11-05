from pydantic import Field

from exo.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.common import CommandId, DeploymentId, NodeId, SystemId
from exo.shared.types.deployments import DeploymentRecord, DeploymentSpec
from exo.shared.types.instance_link import InstanceLinkId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    BrokenInstanceReason,
    Instance,
    InstanceId,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class BaseCommand(TaggedModel):
    command_id: CommandId = Field(default_factory=CommandId)


class TestCommand(BaseCommand):
    __test__ = False


class TextGeneration(BaseCommand):
    task_params: TextGenerationTaskParams


class ImageGeneration(BaseCommand):
    task_params: ImageGenerationTaskParams


class ImageEdits(BaseCommand):
    task_params: ImageEditsTaskParams


class CreateDeployment(BaseCommand):
    """Launch a model, as a deployment the cluster keeps running. Idempotent: a deployment id
    the master already has, or had, changes nothing."""

    deployment_id: DeploymentId
    spec: DeploymentSpec
    model_card: ModelCard
    # Where it runs: the placement the API made, or the preview the user picked. Its
    # deployment_id must be deployment_id.
    instance: Instance


class DeleteDeployment(BaseCommand):
    """A user stops a model."""

    deployment_id: DeploymentId


class ResumeDeployment(BaseCommand):
    """Place a deployment now: clears a hold, or a wait for devices that may come back.

    `revision` is the deployment's when the user resumed it. Resuming bumps it, so a copy of
    the command that arrives again (it is sent until its effect shows) changes nothing."""

    deployment_id: DeploymentId
    revision: int


class AnnounceDeployments(BaseCommand):
    """A node tells the master the deployments it knows of, and the stops. A new master may
    lack them: it started from its own node's copy of the state, or this node was on the other
    side of a split."""

    deployments: list[DeploymentRecord]
    removed: dict[DeploymentId, str]
    # Those this node has only from its disk: no master has run them since it started
    restored: list[DeploymentId]
    # Those its previous master was running: a hold the new master has for one of them is
    # its own side's, and is lifted
    running: list[DeploymentId] = []


class InstanceBroken(BaseCommand):
    """A worker can't run this instance any more and asks for it to be removed. This is not a
    user's stop: the user still wants the model, so it can be started again. The master ignores
    it for an instance that is already gone, so a late one changes nothing."""

    instance_id: InstanceId
    reason: BrokenInstanceReason
    # What went wrong, for the log
    message: str | None = None


class TaskCancelled(BaseCommand):
    cancelled_command_id: CommandId


class TaskFinished(BaseCommand):
    finished_command_id: CommandId


class SendInputChunk(BaseCommand):
    """Command to send an input image chunk (converted to event by master)."""

    chunk: InputImageChunk


class RequestEventLog(BaseCommand):
    since_idx: int
    # Send a snapshot of the current state rather than the events: a node that has just
    # started following this master can't apply them on top of the state it has
    snapshot: bool = False


class StartDownload(BaseCommand):
    target_node_id: NodeId
    shard_metadata: ShardMetadata


class DeleteDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class CancelDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class AddCustomModelCard(BaseCommand):
    model_card: ModelCard


class DeleteCustomModelCard(BaseCommand):
    model_id: ModelId


class SetInstanceLink(BaseCommand):
    link_id: InstanceLinkId
    prefill_instances: list[InstanceId]
    decode_instances: list[InstanceId]


class DeleteInstanceLink(BaseCommand):
    link_id: InstanceLinkId


DownloadCommand = StartDownload | DeleteDownload | CancelDownload


Command = (
    TestCommand
    | RequestEventLog
    | TextGeneration
    | ImageGeneration
    | ImageEdits
    | CreateDeployment
    | DeleteDeployment
    | ResumeDeployment
    | InstanceBroken
    | AnnounceDeployments
    | TaskCancelled
    | TaskFinished
    | SendInputChunk
    | AddCustomModelCard
    | DeleteCustomModelCard
    | SetInstanceLink
    | DeleteInstanceLink
)


class ForwarderCommand(FrozenModel):
    origin: SystemId
    command: Command


class ForwarderDownloadCommand(FrozenModel):
    origin: SystemId
    command: DownloadCommand
