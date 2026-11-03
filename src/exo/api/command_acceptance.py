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
from exo.shared.types.state import State

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
