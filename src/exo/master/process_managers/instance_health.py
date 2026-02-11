from collections.abc import Sequence
from typing import final

from exo.master.reconcile import instance_connections_healthy, instance_runners_failed
from exo.shared.types.events import Event, InstanceDeleted
from exo.shared.types.state import State


@final
class InstanceHealthReconciler:
    """Delete instances whose network connections are broken or whose runners have all failed."""

    async def reconcile(self, state: State) -> Sequence[Event]:
        events: list[Event] = []
        for instance_id, instance in state.instances.items():
            if not instance_connections_healthy(instance, state.topology):
                events.append(
                    InstanceDeleted(
                        instance_id=instance_id,
                        failure_error="Network connection lost",
                    )
                )
                continue

            is_failed, error_message = instance_runners_failed(
                instance, state.runners
            )
            if is_failed:
                events.append(
                    InstanceDeleted(
                        instance_id=instance_id,
                        failure_error=error_message,
                    )
                )
        return events
