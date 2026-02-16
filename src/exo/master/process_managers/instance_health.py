from collections.abc import Sequence
from typing import final

from loguru import logger

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
                    logger.info(
                        f"Instance {instance_id} failed (attempt"
                        f" {mi.consecutive_failures + 1}/{MAX_INSTANCE_RETRIES}),"
                        f" retrying: {error_message}"
                    )
                    events.append(
                        InstanceRetrying(
                            instance_id=instance_id,
                            meta_instance_id=mid,
                            failure_error=error_message or "Runner failed",
                        )
                    )
                else:
                    if mid and mi:
                        logger.warning(
                            f"Instance {instance_id} exceeded retry limit"
                            f" ({MAX_INSTANCE_RETRIES}), deleting:"
                            f" {error_message}"
                        )
                    events.append(
                        InstanceDeleted(
                            instance_id=instance_id,
                            failure_error=error_message,
                        )
                    )
        return events
