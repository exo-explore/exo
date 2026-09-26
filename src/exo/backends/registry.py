"""Route a bound runner to the builder for its placement.

Image placements stay on the mflux builder. Tinygrad placements use
``TinygradBuilder``. Every other placement stays on ``MlxBuilder``, which is
the default engine constructed by the runner entrypoint.
"""

from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance, TinygradInstance
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder


def resolve_builder(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    cancel_receiver: MpReceiver[TaskId],
) -> Builder:
    """Construct the builder for ``bound_instance``.

    Tinygrad device selection happens inside ``TinygradBuilder`` and reads the
    per-node backend stored on ``TinygradInstance``. It does not inspect the
    operating system.

    Raises:
        TinygradDeviceSelectionError: The tinygrad package is missing or the
            instance names a non-tinygrad backend. The runner entrypoint
            handles this by publishing ``RunnerTerminationError``.
    """
    if bound_instance.is_image_model:
        from exo.worker.engines.image.builder import MfluxBuilder

        return MfluxBuilder(
            event_sender,
            cancel_receiver,
            bound_instance.bound_shard,
        )

    instance = bound_instance.instance
    if isinstance(instance, TinygradInstance):
        from exo.backends.tinygrad_engine import (
            TinygradBuilder,
            tinygrad_device_name_for_backend,
        )

        device_name = tinygrad_device_name_for_backend(
            instance.backend_for_node(bound_instance.bound_node_id)
        )
        return TinygradBuilder(
            device_name=device_name,
            model_id=bound_instance.bound_shard.model_card.model_id,
            event_sender=event_sender,
            cancel_receiver=cancel_receiver,
        )

    from exo.worker.engines.mlx.patches import apply_mlx_patches

    apply_mlx_patches()

    from exo.worker.engines.mlx.builder import MlxBuilder

    return MlxBuilder(
        model_id=bound_instance.bound_shard.model_card.model_id,
        event_sender=event_sender,
        cancel_receiver=cancel_receiver,
    )
