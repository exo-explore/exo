from collections.abc import Sequence
from typing import final

from exo.master.reconcile import instance_connections_healthy, instance_runners_failed
from exo.shared.types.events import Event, InstanceDeleted, InstanceRetrying
from exo.shared.types.state import State

MAX_INSTANCE_RETRIES = 3


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
                instance, state.runners, state.node_identities
            )
            if is_failed:
                # Retry within the same instance if backed by a MetaInstance
                mid = instance.meta_instance_id
                mi = state.meta_instances.get(mid) if mid else None
                if mid and mi and mi.consecutive_failures < MAX_INSTANCE_RETRIES:
                    events.append(
                        InstanceRetrying(
                            instance_id=instance_id,
                            meta_instance_id=mid,
                            failure_error=error_message or "Runner failed",
                        )
                    )
                else:
                    events.append(
                        InstanceDeleted(
                            instance_id=instance_id,
                            failure_error=error_message,
                        )
                    )
        return events
