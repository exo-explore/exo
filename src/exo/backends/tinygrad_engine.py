"""Tinygrad adapter for the exo inference engine.

This module is a drop-in ``Engine`` / ``Builder`` pair. It loads one pipeline
shard with ``tinygrad.nn.state`` and runs the assigned Llama, Qwen2, or Qwen3
layers. A single-node shard tokenizes GGUF metadata, samples, and yields
token chunks. A multi-node shard passes hidden states to the next rank and
returns the sampled token to rank 0. ``serve_prefill`` stays unimplemented.

Interface map
-------------

``exo.worker.engines.base.Builder`` (runner startup)
    ``connect``             join the placement before weights are read
    ``load``                stream ``ModelLoadingResponse`` while layers load
    ``build``               return a ``TinygradEngine``
    ``close``               drop builder-owned resources

``exo.worker.engines.base.Engine`` (runner loop)
    ``allocate_weights``    sum selected safetensors bytes, without a device copy
    ``load_model``          realize this shard's safetensors or GGUF weights
    ``embed_token_ids``     embed int32 token ids on the first rank
    ``forward_hidden_state`` run the assigned layers and append the local cache
    ``project_logits``      final norm and language-model head on the last rank
    ``warmup``              compile a throwaway forward pass on this rank
    ``submit``              enqueue one ``TextGeneration``
    ``step``                prefill or decode one token, hopping when sharded
    ``serve_prefill``       serve a disaggregated prefill request (unimplemented)
    ``memory_usage``        report ``MemoryUsage`` for this device
    ``pin_assigned_layers`` drop any other layers before a load
    ``close``               drop the loaded shard and its cache

``exo.shared.types.backends.Backend`` (placement registry)
    ``TinygradAmd``         ``Device.DEFAULT = "AMD"`` (ROCm/HIP)
    ``TinygradMetal``       ``Device.DEFAULT = "METAL"`` (Apple Silicon)
    ``TinygradCuda``        ``Device.DEFAULT = "CUDA"`` (NVIDIA)
    ``TinygradCpu``         ``Device.DEFAULT = "CPU"``

Device names are an explicit input. The host operating system is not read.
"""

from __future__ import annotations

import ctypes
import gc
import secrets
import time
from collections import deque
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, BinaryIO, Literal, NoReturn, Protocol, final, override

from exo.shared.types.backends import Backend
from exo.shared.types.chunks import Chunk, TokenChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    ImageEdits,
    ImageGeneration,
    TaskId,
    TextGeneration,
)
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
    from exo.backends.tinygrad_pipeline import (
        PipelineTokenResult,
        PipelineTransport,
    )
    from exo.backends.tinygrad_tokenizer import GgufTokenizer

type TinygradDeviceName = Literal["AMD", "METAL", "CUDA", "CPU"]


class TaskCancellationReceiver(Protocol):
    """Source of task ids the runner has asked this engine to cancel."""

    def collect(self) -> list[TaskId]: ...


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


def _last_logit_row(logits: Tensor) -> list[float]:
    """Copy the last vocabulary row of ``logits`` to host floats.

    Raises:
        TinygradWeightError: The runner entrypoint handles a logit tensor
            whose rank is not 2 or 3.
    """
    from tinygrad import dtypes

    from exo.backends.tinygrad_weights import TinygradWeightError

    shape = tuple(int(dimension) for dimension in logits.shape)
    if len(shape) == 3:
        sequence = shape[1]
        vocabulary = shape[2]
        row = logits.shrink(
            ((0, 1), (sequence - 1, sequence), (0, vocabulary))
        ).reshape(vocabulary)
    elif len(shape) == 2:
        sequence = shape[0]
        vocabulary = shape[1]
        row = logits.shrink(((sequence - 1, sequence), (0, vocabulary))).reshape(
            vocabulary
        )
    else:
        raise TinygradWeightError(
            f"Logits have shape {shape}, expected a sequence and vocabulary"
        )
    payload = (
        row.float()
        .contiguous()
        .realize()
        .bitcast(dtypes.uint8)
        .contiguous()
        .realize()
        .numpy()
        .tobytes()
    )
    if len(payload) % 4 != 0:
        raise TinygradWeightError("Logit byte length is not a multiple of 4")
    values: list[float] = []
    for offset in range(0, len(payload), 4):
        values.append(
            float(ctypes.c_float.from_buffer_copy(payload[offset : offset + 4]).value)
        )
    return values


def _unimplemented(operation: str) -> NoReturn:
    """Raise the stub error for an operation the runner will call.

    Raises:
        NotImplementedError: Disaggregated prefill is not implemented. The
            runner entrypoint handles this by publishing
            ``RunnerTerminationError``.
    """
    raise NotImplementedError(f"Tinygrad {operation} is not implemented")


@final
@dataclass
class _ActiveGeneration:
    """One text request whose prompt is already encoded."""

    task: TextGeneration
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: list[int]
    generated_text: str
    seed: int
    max_completion_tokens: int
    stop_strings: tuple[str, ...]
    started_at: float
    prefill_seconds: float


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
    model_id: ModelId | None = field(init=False, default=None)
    gguf_tokenizer: GgufTokenizer | None = field(init=False, default=None)
    cancel_receiver: TaskCancellationReceiver | None = field(init=False, default=None)
    pipeline_transport: PipelineTransport | None = field(init=False, default=None)
    _pipeline_shard_metadata: PipelineShardMetadata | None = field(
        init=False, default=None
    )
    _cancelled_tasks: set[TaskId] = field(init=False, default_factory=set)
    _pending_tasks: deque[TextGeneration] = field(init=False, default_factory=deque)
    _active_generation: _ActiveGeneration | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        assign_tinygrad_default_device(self.device_name)

    def memory_usage(self) -> MemoryUsage:
        """Return this device's capacity in the placement ``MemoryUsage`` shape.

        The query does not open the device or copy weights. ``METAL`` and
        ``CPU`` report unified memory through psutil. ``AMD`` and ``CUDA``
        report free accelerator memory. ``InfoGatherer`` publishes the same
        value before this engine exists, and the master weights shards with
        ``ram_available``.
        """
        from exo.backends.tinygrad_memory import memory_usage_for_device

        return memory_usage_for_device(self.device_name)

    def pin_assigned_layers(
        self, bound_instance: BoundInstance
    ) -> PipelineShardMetadata:
        """Drop every realized tensor that is not the commanded layer interval.

        Args:
            bound_instance: Placement whose pipeline shard carries
                ``start_layer`` and ``end_layer``.

        Returns:
            The pipeline shard load will realize. Only that half-open interval
            is allowed onto the device.

        Raises:
            TinygradWeightError: The runner entrypoint handles this when the
                placement is not a pipeline shard.
        """
        shard = _pipeline_shard(bound_instance)
        self._release_loaded_graph()
        return shard

    def _release_loaded_graph(self) -> None:
        self.loaded_shard = None
        self.key_value_cache = None
        self._active_generation = None
        self._pipeline_shard_metadata = None
        gc.collect()

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
        from exo.backends.tinygrad_checkpoint import (
            assert_assigned_tensor_shapes,
            directory_has_gguf,
            load_checkpoint_architecture,
            read_gguf_checkpoint,
        )
        from exo.backends.tinygrad_llama import (
            LocalKeyValueCache,
            assemble_loaded_shard,
        )
        from exo.backends.tinygrad_tokenizer import gguf_tokenizer_from_checkpoint
        from exo.backends.tinygrad_weights import (
            TinygradWeightError,
            iter_realized_parameter_groups,
            model_directory_for_shard,
        )
        from exo.download.huggingface_utils import extract_layer_num

        shard = self.pin_assigned_layers(bound_instance)
        directory = model_directory_for_shard(shard)
        self.model_id = shard.model_card.model_id
        self.gguf_tokenizer = None
        if directory_has_gguf(directory):
            self.gguf_tokenizer = gguf_tokenizer_from_checkpoint(
                read_gguf_checkpoint(directory)
            )
        architecture = load_checkpoint_architecture(directory)
        total_layers = shard.end_layer - shard.start_layer
        collected: dict[str, Tensor] = {}
        layers_loaded = 0
        for group in iter_realized_parameter_groups(shard):
            for tensor_name, tensor in group.parameters:
                collected[tensor_name] = tensor
            if group.layer_index is not None and not (
                shard.start_layer <= group.layer_index < shard.end_layer
            ):
                raise TinygradWeightError(
                    f"Layer {group.layer_index} is outside "
                    f"[{shard.start_layer}, {shard.end_layer})"
                )
            if group.layer_index is None:
                continue
            layers_loaded += 1
            yield ModelLoadingResponse(layers_loaded=layers_loaded, total=total_layers)
        if total_layers == 0:
            yield ModelLoadingResponse(layers_loaded=0, total=0)
        for tensor_name in collected:
            layer_index = extract_layer_num(tensor_name)
            if layer_index is not None and not (
                shard.start_layer <= layer_index < shard.end_layer
            ):
                raise TinygradWeightError(
                    f"Realized tensor {tensor_name} is outside "
                    f"[{shard.start_layer}, {shard.end_layer})"
                )
        assert_assigned_tensor_shapes(collected, architecture, shard)
        self.loaded_shard = assemble_loaded_shard(collected, architecture, shard)
        self.key_value_cache = LocalKeyValueCache(len(self.loaded_shard.layers))
        self._pipeline_shard_metadata = shard

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
        """Compile and run a throwaway forward pass, then drop its cache.

        A multi-rank shard warms only the layers it owns. Nothing is sent to
        the next rank.

        Returns:
            None. The key-value cache is empty when this returns.

        Raises:
            TinygradShardRoleError: The runner entrypoint handles a one-node
                shard that does not contain every layer, or a multi-rank shard
                that is both the first and last layer.
            TinygradWeightError: The runner entrypoint handles a missing
                tokenizer or a forward that runs before the shard is loaded.
        """
        shard = self._require_pipeline_shard()
        if shard.world_size == 1:
            self._require_full_shard()
            _ = self._realize_logits((self._warmup_token_id(),))
            self._require_key_value_cache().clear()
            return
        loaded = self._require_loaded_shard()
        if loaded.is_first_layer and loaded.is_last_layer:
            from exo.backends.tinygrad_weights import TinygradShardRoleError

            raise TinygradShardRoleError(
                "A multi-rank tinygrad pipeline shard cannot contain every layer"
            )
        if loaded.is_first_layer:
            _ = self._hidden_state_from_token_ids((self._warmup_token_id(),))
        else:
            hidden_state = self.forward_hidden_state(self._warmup_hidden_state())
            if loaded.is_last_layer:
                _ = self.project_logits(hidden_state)
        self._require_key_value_cache().clear()

    @override
    def submit(self, task: GenerationTask) -> None:
        """Enqueue one text generation request.

        Args:
            task: A ``TextGeneration`` task. Image tasks are rejected.

        Returns:
            None. The task is retained until ``step`` finishes or cancels it.

        Raises:
            TinygradModelSupportError: The runner entrypoint handles an image
                task or a text task that carries images.
        """
        from exo.backends.tinygrad_weights import TinygradModelSupportError

        if isinstance(task, (ImageGeneration, ImageEdits)):
            raise TinygradModelSupportError(
                "Tinygrad generation serves text GGUF models"
            )
        if task.task_params.images:
            raise TinygradModelSupportError(
                "Tinygrad generation does not accept image inputs"
            )
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._pending_tasks.append(task)

    @override
    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        """Prefill a queued prompt or decode one more token.

        Returns:
            A token chunk, and a ``FinishedResponse`` when that token ends
            the request. A cancelled task yields ``CancelledResponse``.

        Raises:
            TinygradShardRoleError: The runner entrypoint handles a one-node
                shard that does not contain every layer.
            TinygradWeightError: The runner entrypoint handles a missing
                tokenizer or an empty encoding.
            TinygradPipelineError: The runner entrypoint handles a multi-rank
                shard whose peer cannot be reached.
        """
        shard = self._require_pipeline_shard()
        if shard.world_size > 1:
            return self._step_pipeline()
        self._require_full_shard()
        self._collect_cancellations()
        output: list[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]] = []
        kept: deque[TextGeneration] = deque()
        for task in self._pending_tasks:
            if self.should_cancel(task.task_id):
                output.append((task.task_id, CancelledResponse()))
                continue
            kept.append(task)
        self._pending_tasks = kept
        active = self._active_generation
        if active is not None and self.should_cancel(active.task.task_id):
            output.append((active.task.task_id, CancelledResponse()))
            self._drop_active_generation()
            return output
        if self._active_generation is None:
            if not self._pending_tasks:
                return output
            self._active_generation = self._start_generation(
                self._pending_tasks.popleft()
            )
        active = self._active_generation
        chunk, finished = self._advance_generation(active)
        output.append((active.task.task_id, chunk))
        if finished:
            output.append((active.task.task_id, FinishedResponse()))
            self._drop_active_generation()
        return output

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

    def _require_full_shard(self) -> LoadedShard:
        from exo.backends.tinygrad_weights import TinygradShardRoleError

        loaded_shard = self._require_loaded_shard()
        if not loaded_shard.is_first_layer or not loaded_shard.is_last_layer:
            raise TinygradShardRoleError(
                "Tinygrad generation serves a shard that contains every layer"
            )
        return loaded_shard

    def _require_tokenizer(self) -> GgufTokenizer:
        from exo.backends.tinygrad_weights import TinygradWeightError

        tokenizer = self.gguf_tokenizer
        if tokenizer is None:
            raise TinygradWeightError(
                "GGUF tokenizer is missing, so this checkpoint cannot be served"
            )
        return tokenizer

    def _require_model_id(self) -> ModelId:
        from exo.backends.tinygrad_weights import TinygradWeightError

        model_id = self.model_id
        if model_id is None:
            raise TinygradWeightError(
                "Tinygrad generation was called before the model was loaded"
            )
        return model_id

    def _collect_cancellations(self) -> None:
        receiver = self.cancel_receiver
        if receiver is None:
            return
        for task_id in receiver.collect():
            self._cancelled_tasks.add(task_id)

    def _start_generation(self, task: TextGeneration) -> _ActiveGeneration:
        from exo.backends.tinygrad_generate import (
            completion_token_limit,
            normalize_stop_strings,
        )
        from exo.backends.tinygrad_tokenizer import encode_chat
        from exo.backends.tinygrad_weights import TinygradWeightError

        tokenizer = self._require_tokenizer()
        messages = [
            (message.role, str(message.content)) for message in task.task_params.input
        ]
        instructions = task.task_params.instructions
        token_ids = encode_chat(
            tokenizer,
            messages,
            None if instructions is None else str(instructions),
        )
        if not token_ids:
            raise TinygradWeightError("GGUF chat encoding produced no tokens")
        seed = task.task_params.seed
        if seed is None:
            seed = secrets.randbits(32)
        return _ActiveGeneration(
            task=task,
            prompt_token_ids=token_ids,
            generated_token_ids=[],
            generated_text="",
            seed=seed,
            max_completion_tokens=completion_token_limit(
                task.task_params.max_output_tokens
            ),
            stop_strings=normalize_stop_strings(task.task_params.stop),
            started_at=time.perf_counter(),
            prefill_seconds=0.0,
        )

    def _advance_generation(self, active: _ActiveGeneration) -> tuple[TokenChunk, bool]:
        from exo.backends.tinygrad_generate import sample_token_id

        if not active.generated_token_ids:
            logits = self._realize_logits(active.prompt_token_ids)
            active.prefill_seconds = time.perf_counter() - active.started_at
        else:
            logits = self._realize_logits((active.generated_token_ids[-1],))
        token_id = sample_token_id(
            logits,
            temperature=active.task.task_params.temperature,
            top_k=active.task.task_params.top_k,
            top_p=active.task.task_params.top_p,
            seed=active.seed,
            draw_index=len(active.generated_token_ids),
        )
        piece, finish_reason = self._accept_sampled_token(active, token_id)
        return self._token_chunk(active, token_id, piece, finish_reason)

    def _step_pipeline(
        self,
    ) -> list[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        from exo.backends.tinygrad_weights import TinygradShardRoleError

        loaded = self._require_loaded_shard()
        if loaded.is_first_layer and loaded.is_last_layer:
            raise TinygradShardRoleError(
                "A multi-rank tinygrad pipeline shard cannot contain every layer"
            )
        transport = self._require_pipeline_transport()
        self._collect_cancellations()
        output: list[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]] = []
        kept: deque[TextGeneration] = deque()
        for task in self._pending_tasks:
            if self.should_cancel(task.task_id):
                transport.signal_cancellation(task.task_id)
                output.append((task.task_id, CancelledResponse()))
                continue
            kept.append(task)
        self._pending_tasks = kept
        active = self._active_generation
        if active is not None and self.should_cancel(active.task.task_id):
            transport.signal_cancellation(active.task.task_id)
            output.append((active.task.task_id, CancelledResponse()))
            self._drop_active_generation()
            return output
        if self._active_generation is None:
            if not self._pending_tasks:
                return output
            self._active_generation = self._start_generation(
                self._pending_tasks.popleft()
            )
        active = self._active_generation
        task_id = active.task.task_id
        if loaded.is_first_layer:
            hidden_state = self._hidden_state_for_active(active)
            transport.send_hidden_state(task_id, hidden_state)
            result = transport.receive_token_result(task_id)
            if result is None:
                output.append((task_id, CancelledResponse()))
                self._drop_active_generation()
                return output
            chunk, finished = self._chunk_from_remote_token(active, result)
            output.append((task_id, chunk))
        else:
            hidden_state = transport.receive_hidden_state(task_id)
            if hidden_state is None:
                output.append((task_id, CancelledResponse()))
                self._drop_active_generation()
                return output
            hidden_state = self.forward_hidden_state(hidden_state)
            if loaded.is_last_layer:
                result = self._sample_pipeline_token(active, hidden_state)
                transport.send_token_result(task_id, result)
                finished = result.finish_reason is not None
            else:
                transport.send_hidden_state(task_id, hidden_state)
                forwarded = transport.receive_token_result(task_id)
                if forwarded is None:
                    output.append((task_id, CancelledResponse()))
                    self._drop_active_generation()
                    return output
                transport.send_token_result(task_id, forwarded)
                finished = forwarded.finish_reason is not None
        if finished:
            output.append((task_id, FinishedResponse()))
            self._drop_active_generation()
        return output

    def _hidden_state_for_active(self, active: _ActiveGeneration) -> HiddenStateBuffer:
        if not active.generated_token_ids:
            return self._hidden_state_from_token_ids(active.prompt_token_ids)
        return self._hidden_state_from_token_ids((active.generated_token_ids[-1],))

    def _sample_pipeline_token(
        self, active: _ActiveGeneration, hidden_state: HiddenStateBuffer
    ) -> PipelineTokenResult:
        from exo.backends.tinygrad_generate import sample_token_id
        from exo.backends.tinygrad_pipeline import PipelineTokenResult

        if not active.generated_token_ids:
            active.prefill_seconds = time.perf_counter() - active.started_at
        token_id = sample_token_id(
            self._logits_from_hidden_state(hidden_state),
            temperature=active.task.task_params.temperature,
            top_k=active.task.task_params.top_k,
            top_p=active.task.task_params.top_p,
            seed=active.seed,
            draw_index=len(active.generated_token_ids),
        )
        piece, finish_reason = self._accept_sampled_token(active, token_id)
        return PipelineTokenResult(
            token_id=token_id,
            text=piece,
            finish_reason=finish_reason,
            prompt_token_count=len(active.prompt_token_ids),
            completion_token_count=len(active.generated_token_ids),
        )

    def _chunk_from_remote_token(
        self, active: _ActiveGeneration, result: PipelineTokenResult
    ) -> tuple[TokenChunk, bool]:
        if not active.generated_token_ids:
            active.prefill_seconds = time.perf_counter() - active.started_at
        active.generated_token_ids.append(result.token_id)
        active.generated_text += result.text
        return self._token_chunk(
            active,
            result.token_id,
            result.text,
            result.finish_reason,
            prompt_tokens=result.prompt_token_count,
            completion_tokens=result.completion_token_count,
        )

    def _accept_sampled_token(
        self, active: _ActiveGeneration, token_id: int
    ) -> tuple[str, Literal["stop", "length"] | None]:
        from exo.backends.tinygrad_generate import visible_completion_piece
        from exo.backends.tinygrad_tokenizer import incremental_token_text

        tokenizer = self._require_tokenizer()
        if not active.generated_token_ids:
            previous_ids: tuple[int, ...] = ()
        else:
            previous_ids = tuple(active.generated_token_ids)
        piece = incremental_token_text(tokenizer, previous_ids, token_id)
        piece, stop = visible_completion_piece(
            active.generated_text, piece, active.stop_strings
        )
        active.generated_token_ids.append(token_id)
        active.generated_text += piece
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            return "", "stop"
        if stop is not None:
            return piece, "stop"
        if len(active.generated_token_ids) >= active.max_completion_tokens:
            return piece, "length"
        return piece, None

    def _token_chunk(
        self,
        active: _ActiveGeneration,
        token_id: int,
        piece: str,
        finish_reason: Literal["stop", "length"] | None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> tuple[TokenChunk, bool]:
        from exo.backends.tinygrad_generate import generation_stats, generation_usage

        usage = None
        stats = None
        if finish_reason is not None:
            resolved_prompt = (
                len(active.prompt_token_ids) if prompt_tokens is None else prompt_tokens
            )
            resolved_completion = (
                len(active.generated_token_ids)
                if completion_tokens is None
                else completion_tokens
            )
            usage = generation_usage(resolved_prompt, resolved_completion)
            stats = generation_stats(
                prompt_tokens=resolved_prompt,
                completion_tokens=resolved_completion,
                prefill_seconds=active.prefill_seconds,
                generation_seconds=max(
                    time.perf_counter() - active.started_at - active.prefill_seconds,
                    0.0,
                ),
                parameter_byte_count=self.parameter_byte_count,
            )
        chunk = TokenChunk(
            model=self._require_model_id(),
            text=piece,
            token_id=token_id,
            usage=usage,
            finish_reason=finish_reason,
            stats=stats,
        )
        return chunk, finish_reason is not None

    def _require_pipeline_shard(self) -> PipelineShardMetadata:
        from exo.backends.tinygrad_weights import TinygradWeightError

        shard = self._pipeline_shard_metadata
        if shard is None:
            raise TinygradWeightError(
                "Tinygrad generation was called before the model was loaded"
            )
        return shard

    def _require_pipeline_transport(self) -> PipelineTransport:
        from exo.backends.tinygrad_pipeline import TinygradPipelineError

        transport = self.pipeline_transport
        if transport is None:
            raise TinygradPipelineError("Tinygrad pipeline generation has no transport")
        transport.cancellation_probe = self._pipeline_cancellation_requested
        return transport

    def _pipeline_cancellation_requested(self) -> bool:
        self._collect_cancellations()
        if CANCEL_ALL_TASKS in self._cancelled_tasks:
            return True
        active = self._active_generation
        return active is not None and self.should_cancel(active.task.task_id)

    def _warmup_token_id(self) -> int:
        tokenizer = self._require_tokenizer()
        token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
        if token_id < 0 or token_id >= len(tokenizer.tokens):
            return 0
        return token_id

    def _warmup_hidden_state(self) -> HiddenStateBuffer:
        from exo.backends.tinygrad_hidden_state import HiddenStateBuffer

        hidden_size = self._require_loaded_shard().architecture.hidden_size
        return HiddenStateBuffer(
            dtype="float16",
            shape=(1, 1, hidden_size),
            data=b"\x00" * (2 * hidden_size),
        )

    def _hidden_state_from_token_ids(
        self, token_ids: Sequence[int]
    ) -> HiddenStateBuffer:
        from exo.backends.tinygrad_hidden_state import (
            TokenIdBuffer,
            tensor_to_hidden_state,
            token_ids_to_tensor,
        )
        from exo.backends.tinygrad_llama import embed_token_tensor, forward_loaded_shard
        from exo.backends.tinygrad_weights import TinygradWeightError

        if not token_ids:
            raise TinygradWeightError(
                "Tinygrad generation received an empty token sequence"
            )
        shard = self._require_loaded_shard()
        packed = b"".join(
            token_id.to_bytes(4, "little", signed=True) for token_id in token_ids
        )
        hidden = embed_token_tensor(
            shard,
            token_ids_to_tensor(TokenIdBuffer(shape=(1, len(token_ids)), data=packed)),
        )
        return tensor_to_hidden_state(
            forward_loaded_shard(shard, self._require_key_value_cache(), hidden)
        )

    def _logits_from_hidden_state(self, hidden_state: HiddenStateBuffer) -> list[float]:
        from exo.backends.tinygrad_hidden_state import hidden_state_to_tensor
        from exo.backends.tinygrad_llama import project_logits_tensor

        logits = project_logits_tensor(
            self._require_loaded_shard(), hidden_state_to_tensor(hidden_state)
        )
        return _last_logit_row(logits)

    def _realize_logits(self, token_ids: Sequence[int]) -> list[float]:
        from exo.backends.tinygrad_hidden_state import (
            TokenIdBuffer,
            token_ids_to_tensor,
        )
        from exo.backends.tinygrad_llama import (
            embed_token_tensor,
            forward_loaded_shard,
            project_logits_tensor,
        )
        from exo.backends.tinygrad_weights import TinygradWeightError

        if not token_ids:
            raise TinygradWeightError(
                "Tinygrad generation received an empty token sequence"
            )
        shard = self._require_full_shard()
        cache = self._require_key_value_cache()
        packed = b"".join(
            token_id.to_bytes(4, "little", signed=True) for token_id in token_ids
        )
        hidden = embed_token_tensor(
            shard,
            token_ids_to_tensor(TokenIdBuffer(shape=(1, len(token_ids)), data=packed)),
        )
        hidden = forward_loaded_shard(shard, cache, hidden)
        logits = project_logits_tensor(shard, hidden)
        return _last_logit_row(logits)

    def _drop_active_generation(self) -> None:
        self._active_generation = None
        cache = self.key_value_cache
        if cache is not None:
            cache.clear()

    @override
    def close(self) -> None:
        """Drop the loaded shard and its on-device cache.

        Returns:
            None. Device buffers become unreachable for collection.
        """
        self._release_loaded_graph()
        self.parameter_byte_count = None
        self.gguf_tokenizer = None
        self._pending_tasks.clear()
        self._active_generation = None
        self._cancelled_tasks.clear()
        transport = self.pipeline_transport
        self.pipeline_transport = None
        if transport is not None:
            transport.close()


@final
@dataclass
class TinygradBuilder(Builder):
    """Build a ``TinygradEngine`` for one bound runner.

    Construction assigns ``Device.DEFAULT`` before ``connect`` or ``load``
    run, using the device name the placement selected for this node.
    ``connect`` handshakes with the neighbouring ranks when the shard is one
    stage of a multi-node pipeline. A single-node shard does not open a socket.
    """

    device_name: TinygradDeviceName
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    _engine: TinygradEngine | None = field(init=False, default=None)
    _pipeline_transport: PipelineTransport | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        assign_tinygrad_default_device(self.device_name)

    @override
    def connect(self, bound_instance: BoundInstance) -> None:
        """Handshake with the next and previous ranks when the pipeline is split.

        Args:
            bound_instance: This runner's instance, node, and shard.

        Returns:
            None. A one-node shard returns without opening a socket.

        Raises:
            TinygradPipelineError: The runner entrypoint handles a neighbour
                that does not complete the handshake.
            TinygradWeightError: The runner entrypoint handles a placement
                that is not a tinygrad pipeline shard.
        """
        from exo.backends.tinygrad_pipeline import (
            TcpPipelineTransport,
            TinygradPipelineError,
        )
        from exo.backends.tinygrad_weights import TinygradWeightError
        from exo.shared.types.worker.instances import TinygradInstance

        instance = bound_instance.instance
        if not isinstance(instance, TinygradInstance):
            raise TinygradWeightError(
                "Tinygrad connect received a placement that is not tinygrad"
            )
        shard = _pipeline_shard(bound_instance)
        if shard.world_size == 1:
            return
        hosts = instance.hosts_by_node.get(bound_instance.bound_node_id)
        if hosts is None:
            raise TinygradPipelineError(
                f"Tinygrad pipeline rank {shard.device_rank} has no peer addresses"
            )
        transport = TcpPipelineTransport(device_rank=shard.device_rank, hosts=hosts)
        transport.open()
        self._pipeline_transport = transport

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
        transport = self._pipeline_transport
        self._pipeline_transport = None
        if transport is not None:
            engine.pipeline_transport = transport
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
        engine.cancel_receiver = self.cancel_receiver
        if engine.model_id is None:
            engine.model_id = self.model_id
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
            return
        transport = self._pipeline_transport
        self._pipeline_transport = None
        if transport is not None:
            transport.close()
