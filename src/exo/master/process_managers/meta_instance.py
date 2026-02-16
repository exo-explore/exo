from collections.abc import Sequence
from typing import final

import anyio
from loguru import logger

from exo.master.reconcile import (
    find_unsatisfied_meta_instances,
    try_place_for_meta_instance,
)
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.events import Event, InstanceCreated, MetaInstancePlacementFailed
from exo.shared.types.state import State
from exo.shared.types.worker.instances import Instance, InstanceId

MODEL_CARD_LOAD_TIMEOUT_SECONDS = 10


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
            try:
                with anyio.fail_after(MODEL_CARD_LOAD_TIMEOUT_SECONDS):
                    model_card = await ModelCard.load(meta_instance.model_id)
            except TimeoutError:
                logger.warning(
                    f"ModelCard.load timed out for {meta_instance.model_id}, skipping this cycle"
                )
                continue
            except Exception as exc:
                logger.warning(
                    f"ModelCard.load failed for {meta_instance.model_id}: {exc}"
                )
                error = f"Failed to load model card: {exc}"
                if meta_instance.placement_error != error:
                    all_events.append(
                        MetaInstancePlacementFailed(
                            meta_instance_id=meta_instance.meta_instance_id,
                            reason=error,
                        )
                    )
                continue

            result = try_place_for_meta_instance(
                meta_instance,
                model_card,
                state.topology,
                current_instances,
                state.node_memory,
                state.node_network,
            )
            # Update local instance map so next placement sees this one
            for event in result.events:
                if isinstance(event, InstanceCreated):
                    logger.info(
                        f"MetaInstance reconciler placed instance"
                        f" {event.instance.instance_id} for"
                        f" {meta_instance.model_id}"
                    )
                    current_instances[event.instance.instance_id] = event.instance
            all_events.extend(result.events)

            # Emit placement failure if error differs from what's already in state
            if (
                result.error is not None
                and meta_instance.placement_error != result.error
            ):
                logger.warning(
                    f"MetaInstance placement failed for"
                    f" {meta_instance.model_id}: {result.error}"
                )
                all_events.append(
                    MetaInstancePlacementFailed(
                        meta_instance_id=meta_instance.meta_instance_id,
                        reason=result.error,
                    )
                )
        return all_events
