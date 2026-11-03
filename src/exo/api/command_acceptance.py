"""Whether the cluster has applied a command that changes it, as a node's state shows.

The API sends these commands to the master and has no reply to wait for: the master only
publishes the events the command leads to. A command can be lost on its way, dropped under
load or sent to a master that has just died, so the API sends it again until its effect shows
in the state. Applying one of these commands a second time changes nothing, so a command that
arrives twice (once at the old master, once at the new one) has the effect of one.
"""

from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    DeleteInstanceLink,
    SetInstanceLink,
)
from exo.shared.types.common import ModelId
from exo.shared.types.instance_link import InstanceLinkId
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId

# The commands a request that changes the cluster sends: launching or stopping a model,
# linking instances, adding or removing a custom model card
type ClusterCommand = (
    CreateInstance
    | DeleteInstance
    | SetInstanceLink
    | DeleteInstanceLink
    | AddCustomModelCard
    | DeleteCustomModelCard
)


def command_applied(command: ClusterCommand, state: State) -> bool:
    """Whether the state shows the command's effect."""
    match command:
        case CreateInstance(instance=instance):
            return instance.instance_id in state.instances
        case DeleteInstance(instance_id=instance_id):
            return instance_id not in state.instances
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


def changed_by(command: ClusterCommand) -> InstanceId | InstanceLinkId | ModelId:
    """What the command changes: the instance, link or model card it is about."""
    match command:
        case CreateInstance(instance=instance):
            return instance.instance_id
        case DeleteInstance(instance_id=instance_id):
            return instance_id
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
    new instance or link, so a late copy of the first would be a second one. Applying any
    other of these commands late does what trying again does, so it is kept.
    """
    match command:
        case CreateInstance(instance=instance):
            return DeleteInstance(instance_id=instance.instance_id)
        case SetInstanceLink(link_id=link_id) if link_id not in state.instance_links:
            return DeleteInstanceLink(link_id=link_id)
        case _:
            return None
