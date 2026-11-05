"""Whether the cluster has applied a command that changes it, as a node's state shows.

The API sends these commands to the master and has no reply to wait for: the master only
publishes the events the command leads to. A command can be lost on its way, dropped under
load or sent to a master that has just died, so the API sends it again until its effect shows
in the state. Applying one of these commands a second time changes nothing, so a command that
arrives twice (once at the old master, once at the new one) has the effect of one.
"""

from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateDeployment,
    DeleteCustomModelCard,
    DeleteDeployment,
    DeleteInstanceLink,
    ResumeDeployment,
    SetInstanceLink,
)
from exo.shared.types.common import DeploymentId, ModelId
from exo.shared.types.instance_link import InstanceLinkId
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId

# The commands a request that changes the cluster sends: launching, stopping or resuming a
# model, linking instances, adding or removing a custom model card
type ClusterCommand = (
    CreateDeployment
    | DeleteDeployment
    | ResumeDeployment
    | SetInstanceLink
    | DeleteInstanceLink
    | AddCustomModelCard
    | DeleteCustomModelCard
)


def command_applied(command: ClusterCommand, state: State) -> bool:
    """Whether the state shows the command's effect."""
    match command:
        case CreateDeployment(deployment_id=deployment_id, instance=instance):
            # Or stopped already (a launch is never applied again after its stop), or its
            # placement runs already, for a launch of it posted at the same time
            return (
                deployment_id in state.deployments
                or deployment_id in state.removed_deployments
                or instance.instance_id in state.instances
            )
        case DeleteDeployment(deployment_id=deployment_id):
            # Its tombstone, not its absence: a new master that never saw the launch lacks
            # the deployment too, and the stop must still reach it, or a node that has the
            # deployment brings it back
            return deployment_id in state.removed_deployments
        case ResumeDeployment(deployment_id=deployment_id, revision=revision):
            deployment = state.deployments.get(deployment_id)
            return deployment is None or deployment.revision > revision
        case SetInstanceLink():
            link = state.instance_links.get(command.link_id)
            # The master keeps each instance once, in the order given
            return (
                link is not None
                and list(link.prefill_instances)
                == list(dict.fromkeys(command.prefill_instances))
                and list(link.decode_instances)
                == list(dict.fromkeys(command.decode_instances))
            )
        case DeleteInstanceLink(link_id=link_id):
            return link_id not in state.instance_links
        case AddCustomModelCard(model_card=model_card):
            # Present is enough: comparing whole cards would wait forever on a card that
            # changes on its way through the cluster (a field filled in on reading, say)
            return model_card.model_id in state.custom_model_cards
        case DeleteCustomModelCard(model_id=model_id):
            return model_id not in state.custom_model_cards


def command_in_effect(command: ClusterCommand, state: State) -> bool:
    """Whether the state still shows the command's effect: a launch stopped since was
    applied, but is no longer in effect, so it is no longer its request's to restore."""
    match command:
        case CreateDeployment(deployment_id=deployment_id, instance=instance):
            return (
                deployment_id in state.deployments
                or instance.instance_id in state.instances
            )
        case _:
            return command_applied(command, state)


def command_already_applied(command: ClusterCommand, state: State) -> bool:
    """Whether sending the command now would change nothing, so a request repeating a change
    already made (stopping a model that is gone, setting a link as it is) is answered at
    once instead of waiting for a change that will never come."""
    match command:
        case AddCustomModelCard(model_card=model_card):
            # Adding a card again refreshes it: only the very same card is a repeat
            return state.custom_model_cards.get(model_card.model_id) == model_card
        case _:
            return command_applied(command, state)


def changed_by(
    command: ClusterCommand,
) -> DeploymentId | InstanceId | InstanceLinkId | ModelId:
    """What the command changes: the placement, deployment, link or model card it is about.

    A launch of a placement is about that placement (its instance): each request for one
    makes a new deployment, so a client trying the very same placement again after a 503
    takes over from the failed request by it."""
    match command:
        case CreateDeployment(instance=instance):
            return instance.instance_id
        case (
            DeleteDeployment(deployment_id=deployment_id)
            | ResumeDeployment(deployment_id=deployment_id)
        ):
            return deployment_id
        case SetInstanceLink(link_id=link_id) | DeleteInstanceLink(link_id=link_id):
            return link_id
        case AddCustomModelCard(model_card=model_card):
            return model_card.model_id
        case DeleteCustomModelCard(model_id=model_id):
            return model_id


def undoing_command(command: ClusterCommand, state: State) -> ClusterCommand | None:
    """The command that takes `command` back, should it still apply after its request was
    answered as failed; `state` is the state when it was sent.

    Only launches and new links are undone: a client trying again after the failure gets a
    new deployment or link, so a late copy of the first would be a second one. Applying any
    other of these commands late does what trying again does, so it is kept.
    """
    match command:
        case CreateDeployment(deployment_id=deployment_id):
            return DeleteDeployment(deployment_id=deployment_id)
        case SetInstanceLink(link_id=link_id) if link_id not in state.instance_links:
            return DeleteInstanceLink(link_id=link_id)
        case _:
            return None
