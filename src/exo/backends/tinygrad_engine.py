"""Tinygrad adapter for the exo inference engine.

This module is a drop-in ``Engine`` / ``Builder`` pair. Tensor execution is
intentionally unimplemented.

Interface map
-------------

``exo.worker.engines.base.Builder`` (runner startup)
    ``connect``             join the placement before weights are read
    ``load``                stream ``ModelLoadingResponse`` while layers load
    ``build``               return a ``TinygradEngine``
    ``close``               drop builder-owned resources

``exo.worker.engines.base.Engine`` (runner loop)
    ``allocate_weights``    reserve parameter memory for this shard
    ``load_model``          read the shard's weights onto the selected device
    ``warmup``              compile a throwaway forward pass
    ``submit``              enqueue one ``GenerationTask``
    ``step``                run one tensor generation step and yield chunks
    ``serve_prefill``       serve a disaggregated prefill request
    ``close``               release device memory

``exo.shared.types.backends.Backend`` (placement registry)
    ``TinygradAmd``         ``Device.DEFAULT = "AMD"`` (ROCm/HIP)
    ``TinygradMetal``       ``Device.DEFAULT = "METAL"`` (Apple Silicon)
    ``TinygradCuda``        ``Device.DEFAULT = "CUDA"`` (NVIDIA)
    ``TinygradCpu``         ``Device.DEFAULT = "CPU"``

Device names are an explicit input. The host operating system is not read.
"""

from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import BinaryIO, Literal, NoReturn, final, override

from exo.shared.types.backends import Backend
from exo.shared.types.chunks import Chunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import GenerationTask, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Builder, Engine

type TinygradDeviceName = Literal["AMD", "METAL", "CUDA", "CPU"]

TINYGRAD_DEVICE_NAME_BY_BACKEND: Mapping[Backend, TinygradDeviceName] = (
    MappingProxyType(
        {
            Backend.TinygradAmd: "AMD",
            Backend.TinygradMetal: "METAL",
            Backend.TinygradCuda: "CUDA",
            Backend.TinygradCpu: "CPU",
        }
    )
)


@final
class TinygradDeviceSelectionError(Exception):
    """Raised when a tinygrad device name cannot be applied.

    ``exo.worker.runner.bootstrap.entrypoint`` handles this by publishing a
    ``RunnerTerminationError`` and exiting the runner. ``NodeBackends.gather``
    lets the same error propagate so a misdeclared accelerator is not
    advertised as available.
    """


def tinygrad_device_name_for_backend(backend: Backend) -> TinygradDeviceName:
    """Return the tinygrad device name for an exo backend identity.

    Raises:
        TinygradDeviceSelectionError: ``backend`` is not a tinygrad backend.
            Callers in the runner registry handle this via the runner
            entrypoint, which treats it as a failed engine start.
    """
    try:
        return TINYGRAD_DEVICE_NAME_BY_BACKEND[backend]
    except KeyError as error:
        raise TinygradDeviceSelectionError(
            f"{backend.value} is not a tinygrad backend"
        ) from error


def backends_from_declared_devices(device_names: Iterable[str]) -> list[Backend]:
    """Map explicit tinygrad device names to backend identities.

    Blank entries are ignored. Names are matched exactly after stripping
    whitespace, so ``"AMD"`` is ROCm/HIP and an operating-system name is
    rejected.

    Raises:
        TinygradDeviceSelectionError: A name is not one of ``AMD``, ``METAL``,
            ``CUDA``, or ``CPU``. ``NodeBackends.gather`` lets this propagate
            so startup fails instead of advertising an unknown device.
    """
    backends: list[Backend] = []
    backend_by_device_name = {
        device_name: backend
        for backend, device_name in TINYGRAD_DEVICE_NAME_BY_BACKEND.items()
    }
    for device_name in device_names:
        stripped_name = device_name.strip()
        if stripped_name == "":
            continue
        backend = backend_by_device_name.get(stripped_name)
        if backend is None:
            raise TinygradDeviceSelectionError(
                f"Unknown tinygrad device {stripped_name!r}. "
                "Expected one of AMD, METAL, CUDA, CPU."
            )
        backends.append(backend)
    return backends


def assign_tinygrad_default_device(
    device_name: TinygradDeviceName,
) -> TinygradDeviceName:
    """Assign ``tinygrad.Device.DEFAULT`` from an explicit device name.

    ``"AMD"`` selects ROCm/HIP, including an RX 9700 XT. ``"METAL"`` selects
    Apple Silicon. ``"CUDA"`` selects NVIDIA. ``"CPU"`` selects the reference
    device. The host operating system is not read.

    Returns:
        The device name written to ``Device.DEFAULT``.

    Raises:
        TinygradDeviceSelectionError: Transformed from ``ImportError`` when
            the optional tinygrad package is absent. The runner entrypoint
            handles it by publishing ``RunnerTerminationError``.
    """
    try:
        from tinygrad import Device  # pyright: ignore[reportMissingModuleSource]
    except ImportError as error:
        raise TinygradDeviceSelectionError(
            "tinygrad is not installed, so Device.DEFAULT cannot be selected"
        ) from error
    Device.DEFAULT = device_name
    return device_name


def _unimplemented(operation: str) -> NoReturn:
    """Raise the stub error for an operation the runner will call.

    Raises:
        NotImplementedError: Tensor logic is not implemented. The runner
            entrypoint handles this by publishing ``RunnerTerminationError``.
    """
    raise NotImplementedError(
        f"Tinygrad {operation} is not implemented. "
        "Tensor execution is intentionally left for a later change."
    )


@final
@dataclass
class TinygradEngine(Engine):
    """Inference engine that will execute a shard with tinygrad.

    Construction selects the device by assigning ``Device.DEFAULT``. Methods
    that touch tensors raise ``NotImplementedError``.
    """

    device_name: TinygradDeviceName
    _cancelled_tasks: set[TaskId] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        assign_tinygrad_default_device(self.device_name)

    def allocate_weights(self, bound_instance: BoundInstance) -> None:
        """Reserve device memory for the parameters owned by this shard.

        Args:
            bound_instance: Placement whose ``bound_shard`` carries
                ``start_layer``, ``end_layer``, ``device_rank``, and the
                model card. Those fields determine the parameter set.

        Returns:
            None. The reservation stays on this engine until ``close``.
        """
        _unimplemented(f"allocate_weights for {bound_instance.bound_node_id}")

    def load_model(self, bound_instance: BoundInstance) -> None:
        """Read this shard's weights onto the selected tinygrad device.

        Args:
            bound_instance: Placement whose model card identifies the
                checkpoint and whose shard metadata identifies the layer
                interval to materialize.

        Returns:
            None. Loaded parameters stay on this engine for ``step``.
        """
        _unimplemented(
            f"load_model for {bound_instance.bound_shard.model_card.model_id}"
        )

    @override
    def warmup(self) -> None:
        """Compile and run a throwaway forward pass on the selected device.

        Returns:
            None.
        """
        _unimplemented("warmup")

    @override
    def submit(self, task: GenerationTask) -> None:
        """Enqueue one generation request.

        Args:
            task: A ``TextGeneration``, ``ImageGeneration``, or ``ImageEdits``
                task. The task id is the key later yielded by ``step``.

        Returns:
            None. The task is retained until ``step`` finishes or cancels it.
        """
        _unimplemented(f"submit for {task.task_id}")

    @override
    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        """Run one tensor generation step for the queued tasks.

        Returns:
            Zero or more ``(task_id, payload)`` pairs. ``payload`` is a
            ``Chunk`` (token, tool call, image, or error), a
            ``CancelledResponse``, or a ``FinishedResponse``.
        """
        _unimplemented("step")

    @override
    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        """Serve one disaggregated prefill request on this shard.

        Args:
            request: Prefill job carrying ``request_id``, ``model_id``,
                ``token_ids``, and ``start_pos``.
            wfile: Binary stream that receives the prefill response frames.

        Returns:
            None. Response bytes are written to ``wfile``.
        """
        _unimplemented(f"serve_prefill for {request.request_id} via {wfile!r}")

    @override
    def close(self) -> None:
        """Release device memory held by this engine.

        Returns:
            None.
        """
        _unimplemented("engine close")


@final
@dataclass
class TinygradBuilder(Builder):
    """Build a ``TinygradEngine`` for one bound runner.

    Construction assigns ``Device.DEFAULT`` before ``connect`` or ``load``
    run, using the device name the placement selected for this node.
    """

    device_name: TinygradDeviceName
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]

    def __post_init__(self) -> None:
        assign_tinygrad_default_device(self.device_name)

    @override
    def connect(self, bound_instance: BoundInstance) -> None:
        """Join the tinygrad placement group for ``bound_instance``.

        Args:
            bound_instance: This runner's instance, node, and shard.

        Returns:
            None. A later implementation retains the connected group on the
            builder.
        """
        _unimplemented(f"connect for {bound_instance.instance.instance_id}")

    @override
    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        """Load weights, yielding progress for the runner status channel.

        Args:
            bound_instance: Shard to load. ``allocate_weights`` then
                ``load_model`` are the intended callees.

        Yields:
            ``ModelLoadingResponse`` with ``layers_loaded`` and ``total``
            after each layer group is resident on device.
        """
        _unimplemented(f"load for {bound_instance.bound_shard.model_card.model_id}")

    @override
    def build(self) -> Engine:
        """Return the engine for the loaded shard.

        Returns:
            A ``TinygradEngine`` bound to ``device_name``. The runner then
            calls ``warmup``, ``submit``, and ``step`` on that object.
        """
        _unimplemented(f"build for {self.model_id}")

    @override
    def close(self) -> None:
        """Drop model and group resources held by the builder.

        Returns:
            None.
        """
        _unimplemented("builder close")
