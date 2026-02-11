from collections.abc import Sequence
from typing import final

from exo.master.reconcile import (
    find_unsatisfied_meta_instances,
    try_place_for_meta_instance,
)
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.events import Event, InstanceCreated
from exo.shared.types.state import State
from exo.shared.types.worker.instances import Instance, InstanceId


@final
class MetaInstanceReconciler:
    """Place instances for unsatisfied MetaInstances."""

    async def reconcile(self, state: State) -> Sequence[Event]:
        all_events: list[Event] = []
        # Local copy for intermediate tracking — so placement of B
        # sees A's instance and doesn't double-place on same resources.
        current_instances: dict[InstanceId, Instance] = dict(state.instances)

        unsatisfied = find_unsatisfied_meta_instances(
            state.meta_instances,
            current_instances,
            state.topology,
        )
        for meta_instance in unsatisfied:
            model_card = await ModelCard.load(meta_instance.model_id)
            events = try_place_for_meta_instance(
                meta_instance,
                model_card,
                state.topology,
                current_instances,
                state.node_memory,
                state.node_network,
            )
            # Update local instance map so next placement sees this one
            for event in events:
                if isinstance(event, InstanceCreated):
                    current_instances[event.instance.instance_id] = event.instance
            all_events.extend(events)
        return all_events
