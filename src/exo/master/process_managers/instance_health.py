from collections.abc import Sequence
from typing import final

from exo.master.reconcile import instance_connections_healthy
from exo.shared.types.events import Event, InstanceDeleted
from exo.shared.types.state import State


@final
class InstanceHealthReconciler:
    """Delete instances whose network connections are broken."""

    async def reconcile(self, state: State) -> Sequence[Event]:
        return [
            InstanceDeleted(instance_id=instance_id)
            for instance_id, instance in state.instances.items()
            if not instance_connections_healthy(instance, state.topology)
        ]
