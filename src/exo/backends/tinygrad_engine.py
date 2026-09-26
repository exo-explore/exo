"""Tinygrad adapter for the exo inference engine.

This module is a drop-in ``Engine`` / ``Builder`` pair. It loads one pipeline
shard with ``tinygrad.nn.state`` and runs the assigned Llama, Qwen2, or Qwen3
layers. Exo still owns tokenization, sampling, and the network hop. ``warmup``,
``submit``, ``step``, and ``serve_prefill`` stay unimplemented.

Interface map
-------------

``exo.worker.engines.base.Builder`` (runner startup)
    ``connect``             join the placement before weights are read
    ``load``                stream ``ModelLoadingResponse`` while layers load
    ``build``               return a ``TinygradEngine``
    ``close``               drop builder-owned resources

``exo.worker.engines.base.Engine`` (runner loop)
    ``allocate_weights``    sum selected safetensors bytes, without a device copy
    ``load_model``          realize this shard's weights on ``Device.DEFAULT``
    ``embed_token_ids``     embed int32 token ids on the first rank
    ``forward_hidden_state`` run the assigned layers and append the local cache
    ``project_logits``      final norm and language-model head on the last rank
    ``warmup``              compile a throwaway forward pass (unimplemented)
    ``submit``              enqueue one ``GenerationTask`` (unimplemented)
    ``step``                run one tensor generation step (unimplemented)
    ``serve_prefill``       serve a disaggregated prefill request (unimplemented)
    ``close``               drop the loaded shard and its cache

``exo.shared.types.backends.Backend`` (placement registry)
    ``TinygradAmd``         ``Device.DEFAULT = "AMD"`` (ROCm/HIP)
    ``TinygradMetal``       ``Device.DEFAULT = "METAL"`` (Apple Silicon)
    ``TinygradCuda``        ``Device.DEFAULT = "CUDA"`` (NVIDIA)
    ``TinygradCpu``         ``Device.DEFAULT = "CPU"``

Device names are an explicit input. The host operating system is not read.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, BinaryIO, Literal, NoReturn, final, override

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
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Builder, Engine

if TYPE_CHECKING:
    from tinygrad.tensor import Tensor

    from exo.backends.tinygrad_hidden_state import HiddenStateBuffer, TokenIdBuffer
    from exo.backends.tinygrad_llama import LoadedShard, LocalKeyValueCache

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
        from tinygrad import Device
    except ImportError as error:
        raise TinygradDeviceSelectionError(
            "tinygrad is not installed, so Device.DEFAULT cannot be selected"
        ) from error
    try:
        Device.DEFAULT = device_name
    except AttributeError:
        # Tinygrad 0.14 rejects writes to Device.DEFAULT. DEV.value is the
        # supported assignment, and Device.DEFAULT reads it back.
        from tinygrad.helpers import DEV

        DEV.value = device_name
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


def _pipeline_shard(bound_instance: BoundInstance) -> PipelineShardMetadata:
    from exo.backends.tinygrad_weights import TinygradWeightError

    shard = bound_instance.bound_shard
    if isinstance(shard, PipelineShardMetadata):
        return shard
    raise TinygradWeightError(
        f"Tinygrad loading requires a pipeline shard, received {type(shard).__name__}"
    )


@final
@dataclass
class TinygradEngine(Engine):
    """Inference engine that executes one pipeline shard with tinygrad.

    Construction selects the device by assigning ``Device.DEFAULT``. The local
    key-value cache stays on that device. Hidden-state bytes are the only
    values that cross a rank boundary.
    """

    device_name: TinygradDeviceName
    parameter_byte_count: int | None = field(init=False, default=None)
    loaded_shard: LoadedShard | None = field(init=False, default=None)
    key_value_cache: LocalKeyValueCache | None = field(init=False, default=None)
    _cancelled_tasks: set[TaskId] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        assign_tinygrad_default_device(self.device_name)

    def allocate_weights(self, bound_instance: BoundInstance) -> None:
        """Sum the safetensors bytes this shard will realize.

        Args:
            bound_instance: Placement whose pipeline shard selects the layer
                interval. Headers are read and parameter data is not copied
                to the device.

        Returns:
            None. ``parameter_byte_count`` holds the sum until ``close``.

        Raises:
            TinygradWeightError: The runner entrypoint handles a missing
                checkpoint or a non-pipeline shard.
            TinygradModelSupportError: The runner entrypoint handles an
                unsupported ``model_type``.
        """
        from exo.backends.tinygrad_weights import shard_parameter_byte_count

        self.parameter_byte_count = shard_parameter_byte_count(
            _pipeline_shard(bound_instance)
        )

    def load_model(self, bound_instance: BoundInstance) -> None:
        """Realize this shard's weights onto the selected tinygrad device.

        Args:
            bound_instance: Placement whose model card identifies the
                checkpoint and whose shard metadata identifies the layer
                interval to materialize.

        Returns:
            None. Loaded parameters stay on this engine.

        Raises:
            TinygradWeightError: The runner entrypoint handles a missing
                weight or a checkpoint that cannot be read.
            TinygradModelSupportError: The runner entrypoint handles an
                unsupported model.
        """
        for _progress in self.iter_load_model(bound_instance):
            pass

    def iter_load_model(
        self, bound_instance: BoundInstance
    ) -> Generator[ModelLoadingResponse]:
        """Realize the shard, yielding one progress value per decoder layer.

        Raises:
            TinygradWeightError: The runner entrypoint handles a missing
                weight or a checkpoint that cannot be read.
            TinygradModelSupportError: The runner entrypoint handles an
                unsupported model.
        """
        from exo.backends.tinygrad_llama import (
            LocalKeyValueCache,
            assemble_loaded_shard,
        )
        from exo.backends.tinygrad_weights import (
            iter_realized_parameter_groups,
            load_architecture,
            model_directory_for_shard,
        )

        shard = _pipeline_shard(bound_instance)
        architecture = load_architecture(model_directory_for_shard(shard))
        total_layers = shard.end_layer - shard.start_layer
        collected: dict[str, Tensor] = {}
        layers_loaded = 0
        for group in iter_realized_parameter_groups(shard):
            for tensor_name, tensor in group.parameters:
                collected[tensor_name] = tensor
            if group.layer_index is None:
                continue
            layers_loaded += 1
            yield ModelLoadingResponse(layers_loaded=layers_loaded, total=total_layers)
        if total_layers == 0:
            yield ModelLoadingResponse(layers_loaded=0, total=0)
        self.loaded_shard = assemble_loaded_shard(collected, architecture, shard)
        self.key_value_cache = LocalKeyValueCache(len(self.loaded_shard.layers))

    def _require_loaded_shard(self) -> LoadedShard:
        from exo.backends.tinygrad_weights import TinygradWeightError

        loaded_shard = self.loaded_shard
        if loaded_shard is None:
            raise TinygradWeightError("Tinygrad forward was called before load_model")
        return loaded_shard

    def _require_key_value_cache(self) -> LocalKeyValueCache:
        from exo.backends.tinygrad_weights import TinygradWeightError

        cache = self.key_value_cache
        if cache is None:
            raise TinygradWeightError("Tinygrad forward was called before load_model")
        return cache

    def embed_token_ids(self, token_ids: TokenIdBuffer) -> HiddenStateBuffer:
        """Embed token ids on the first rank and return the hidden state.

        Raises:
            TinygradShardRoleError: The runner entrypoint handles this when a
                later rank requests embeddings.
            TinygradWeightError: The runner entrypoint handles a missing table
                or a buffer whose length does not match its shape.
        """
        from exo.backends.tinygrad_hidden_state import (
            tensor_to_hidden_state,
            token_ids_to_tensor,
        )
        from exo.backends.tinygrad_llama import embed_token_tensor

        hidden = embed_token_tensor(
            self._require_loaded_shard(), token_ids_to_tensor(token_ids)
        )
        return tensor_to_hidden_state(hidden)

    def forward_hidden_state(
        self, hidden_state: HiddenStateBuffer
    ) -> HiddenStateBuffer:
        """Run this rank's layers and append the on-device key-value cache.

        Raises:
            TinygradWeightError: The runner entrypoint handles a buffer length
                mismatch or a forward that runs before the shard is loaded.
            TinygradModelSupportError: The runner entrypoint handles a Qwen3
                layer that is missing query or key normalization.
        """
        from exo.backends.tinygrad_hidden_state import (
            hidden_state_to_tensor,
            tensor_to_hidden_state,
        )
        from exo.backends.tinygrad_llama import forward_loaded_shard

        updated = forward_loaded_shard(
            self._require_loaded_shard(),
            self._require_key_value_cache(),
            hidden_state_to_tensor(hidden_state),
        )
        return tensor_to_hidden_state(updated)

    def project_logits(self, hidden_state: HiddenStateBuffer) -> HiddenStateBuffer:
        """Project hidden states to logits on the last rank, after the final norm.

        Raises:
            TinygradShardRoleError: The runner entrypoint handles this when an
                earlier rank requests logits.
            TinygradWeightError: The runner entrypoint handles a missing head
                or a buffer length mismatch.
        """
        from exo.backends.tinygrad_hidden_state import (
            hidden_state_to_tensor,
            tensor_to_hidden_state,
        )
        from exo.backends.tinygrad_llama import project_logits_tensor

        logits = project_logits_tensor(
            self._require_loaded_shard(), hidden_state_to_tensor(hidden_state)
        )
        return tensor_to_hidden_state(logits)

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
        """Drop the loaded shard and its on-device cache.

        Returns:
            None. Device buffers become unreachable for collection.
        """
        self.loaded_shard = None
        self.key_value_cache = None
        self.parameter_byte_count = None


@final
@dataclass
class TinygradBuilder(Builder):
    """Build a ``TinygradEngine`` for one bound runner.

    Construction assigns ``Device.DEFAULT`` before ``connect`` or ``load``
    run, using the device name the placement selected for this node. Exo owns
    the network, so ``connect`` does not open a transport.
    """

    device_name: TinygradDeviceName
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    _engine: TinygradEngine | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        assign_tinygrad_default_device(self.device_name)

    @override
    def connect(self, bound_instance: BoundInstance) -> None:
        """Accept the placement. The network hop stays outside this adapter.

        Args:
            bound_instance: This runner's instance, node, and shard.

        Returns:
            None.
        """
        _ = bound_instance.instance.instance_id

    @override
    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        """Load weights, yielding progress for the runner status channel.

        Args:
            bound_instance: Shard to load. ``allocate_weights`` runs first,
                then each decoder layer is realized.

        Yields:
            ``ModelLoadingResponse`` with ``layers_loaded`` and ``total``
            after each decoder layer is resident on device.

        Raises:
            TinygradWeightError: The runner entrypoint handles a missing
                checkpoint. ``build`` also raises it when ``load`` did not
                finish.
        """
        engine = TinygradEngine(device_name=self.device_name)
        engine.allocate_weights(bound_instance)
        yield from engine.iter_load_model(bound_instance)
        self._engine = engine

    @override
    def build(self) -> Engine:
        """Return the engine for the loaded shard.

        Returns:
            The ``TinygradEngine`` produced by ``load``.

        Raises:
            TinygradWeightError: The runner entrypoint handles this when
                ``build`` runs before ``load`` finishes.
        """
        from exo.backends.tinygrad_weights import TinygradWeightError

        engine = self._engine
        if engine is None or engine.loaded_shard is None:
            raise TinygradWeightError(
                f"Tinygrad build for {self.model_id} was called before load finished"
            )
        return engine

    @override
    def close(self) -> None:
        """Drop the engine held by the builder.

        Returns:
            None.
        """
        engine = self._engine
        self._engine = None
        if engine is not None:
            engine.close()
