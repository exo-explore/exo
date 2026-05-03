"""Apple Neural Engine compute and streaming bandwidth probe.

The GPU profiler uses MLX workloads because MLX can directly target Metal.
There is no public ANE Python runtime, so this probe follows the same private
AppleNeuralEngine.framework path used by the reference ANE inference work:
build a tiny MIL program, compile it in memory, bind IOSurface inputs/outputs,
then time `evaluateWithQoS`.

The probe is optional. On non-Darwin hosts, Intel Macs, or machines where the
private ANE classes are unavailable, `measure()` returns None.
"""

import ctypes
import ctypes.util
import os
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, NamedTuple, Self, cast, final

import numpy as np
import numpy.typing as npt
from anyio import to_thread

from exo.utils.pydantic_ext import FrozenModel, TaggedModel

type Pointer = int
type Float16Array = npt.NDArray[np.float16]
type Float32Array = npt.NDArray[np.float32]
type NumericArray = Float16Array | Float32Array
type UInt8Array = npt.NDArray[np.uint8]

_FP16_BYTES_PER_ELEMENT = 2
_FP32_BYTES_PER_ELEMENT = 4

type AnePrecisionBits = Literal[32, 16, 8, 4]
_ANE_PRECISION_BITS: tuple[AnePrecisionBits, ...] = (32, 16, 8, 4)

_COMPUTE_IN_DIM = 2048
_COMPUTE_OUT_DIM = 2048
_COMPUTE_SPATIAL = 768
_COMPUTE_ITERATIONS_PER_PASS = 12
_COMPUTE_PARALLEL_INSTANCES = 2
_NATIVE_QUANTIZATION_SPEEDUP_THRESHOLD = 1.2

_STREAM_CHANNELS = 4096
_STREAM_SPATIAL = 2048
_STREAM_ITERATIONS_PER_PASS = 16

_WARMUP_SECONDS = 0.5
_MEASUREMENT_PASSES = 5
_QOS_USER_INITIATED = 21

_objc: ctypes.CDLL | None = None
_objc_msg_send_address: Pointer | None = None
_iosurface: ctypes.CDLL | None = None

_k_iosurface_width: Pointer | None = None
_k_iosurface_height: Pointer | None = None
_k_iosurface_bytes_per_element: Pointer | None = None
_k_iosurface_bytes_per_row: Pointer | None = None
_k_iosurface_alloc_size: Pointer | None = None
_k_iosurface_pixel_format: Pointer | None = None

_ane_loaded = False
_ane_descriptor_class: Pointer | None = None
_ane_in_memory_model_class: Pointer | None = None
_ane_request_class: Pointer | None = None
_ane_io_surface_object_class: Pointer | None = None


@final
class AnePrecisionProfile(FrozenModel):
    """Per-precision ANE compute and memory-bound probe result."""

    precision_bits: AnePrecisionBits
    weight_bits: AnePrecisionBits
    activation_bits: AnePrecisionBits
    supported: bool
    compute_tops: float | None = None
    weight_only_compute_tops: float | None = None
    single_instance_compute_tops: float | None = None
    compute_instances: int = 1
    memory_bandwidth_gbps: float | None = None
    activation_quantization_speedup: float | None = None
    native_quantized_compute: bool | None = None
    error: str | None = None


@final
class AneProfile(TaggedModel):
    """Wire format for a measured ANE profile, gathered locally on a node."""

    engine: Literal["ane"]
    precision_profiles: Sequence[AnePrecisionProfile]

    @classmethod
    async def measure(cls) -> Self | None:
        if sys.platform != "darwin" or not ane_available():
            return None
        return await to_thread.run_sync(cls._measure_blocking)

    @classmethod
    def _measure_blocking(cls) -> Self:
        rng = np.random.default_rng(0)
        profiles = tuple(
            _measure_precision_profile(bits, rng) for bits in _ANE_PRECISION_BITS
        )
        return cls(engine="ane", precision_profiles=profiles)


class _ComputeMeasurement(NamedTuple):
    tops: float
    single_instance_tops: float
    instances: int


def _measure_precision_profile(
    precision_bits: AnePrecisionBits, rng: np.random.Generator
) -> AnePrecisionProfile:
    try:
        return _measure_supported_precision_profile(precision_bits, rng)
    except Exception as exc:
        return AnePrecisionProfile(
            precision_bits=precision_bits,
            weight_bits=precision_bits,
            activation_bits=8 if precision_bits in (8, 4) else precision_bits,
            supported=False,
            error=_short_error(exc),
        )


def _measure_supported_precision_profile(
    precision_bits: AnePrecisionBits, rng: np.random.Generator
) -> AnePrecisionProfile:
    # Core ML accepts int4 weights here, but the ANE activation Q/DQ path that
    # triggers native low-precision conv uses int8 activations. int4 activation
    # Q/DQ does not compile through this MIL/private-ANE path, so 4-bit means W4A8.
    activation_bits: AnePrecisionBits = (
        8 if precision_bits in (8, 4) else precision_bits
    )
    baseline_measurement: _ComputeMeasurement | None = None

    if precision_bits == 32:
        compute_weights = rng.standard_normal(
            (_COMPUTE_OUT_DIM, _COMPUTE_IN_DIM)
        ).astype(np.float32)
        compute_input = rng.standard_normal((_COMPUTE_IN_DIM, _COMPUTE_SPATIAL)).astype(
            np.float32
        )
        compute_kernels = tuple(
            _compile_static_conv_kernel(
                weights=compute_weights,
                input_channels=_COMPUTE_IN_DIM,
                output_channels=_COMPUTE_OUT_DIM,
                spatial=_COMPUTE_SPATIAL,
                precision_bits=precision_bits,
            )
            for _ in range(_COMPUTE_PARALLEL_INSTANCES)
        )
        for kernel in compute_kernels:
            kernel.write_input(0, compute_input)

        stream_input = rng.standard_normal((_STREAM_CHANNELS, _STREAM_SPATIAL)).astype(
            np.float32
        )
        stream_kernel = _compile_relu_stream_kernel(
            channels=_STREAM_CHANNELS,
            spatial=_STREAM_SPATIAL,
            precision_bits=precision_bits,
        )
        stream_kernel.write_input(0, stream_input)
    elif precision_bits == 16:
        compute_weights = rng.standard_normal(
            (_COMPUTE_OUT_DIM, _COMPUTE_IN_DIM)
        ).astype(np.float16)
        compute_input = rng.standard_normal((_COMPUTE_IN_DIM, _COMPUTE_SPATIAL)).astype(
            np.float16
        )

        stream_input = rng.standard_normal((_STREAM_CHANNELS, _STREAM_SPATIAL)).astype(
            np.float16
        )

        compute_kernels = tuple(
            _compile_static_conv_kernel(
                weights=compute_weights,
                input_channels=_COMPUTE_IN_DIM,
                output_channels=_COMPUTE_OUT_DIM,
                spatial=_COMPUTE_SPATIAL,
                precision_bits=precision_bits,
            )
            for _ in range(_COMPUTE_PARALLEL_INSTANCES)
        )
        for kernel in compute_kernels:
            kernel.write_input(0, compute_input)

        stream_kernel = _compile_relu_stream_kernel(
            channels=_STREAM_CHANNELS,
            spatial=_STREAM_SPATIAL,
            precision_bits=precision_bits,
        )
        stream_kernel.write_input(0, stream_input)
    else:
        compute_weights = rng.standard_normal(
            (_COMPUTE_OUT_DIM, _COMPUTE_IN_DIM)
        ).astype(np.float32)
        compute_input = rng.standard_normal((_COMPUTE_IN_DIM, _COMPUTE_SPATIAL)).astype(
            np.float16
        )

        stream_input = rng.standard_normal((_STREAM_CHANNELS, _STREAM_SPATIAL)).astype(
            np.float16
        )

        compute_kernels = tuple(
            _compile_quantized_static_conv_kernel(
                weights=compute_weights,
                input_channels=_COMPUTE_IN_DIM,
                output_channels=_COMPUTE_OUT_DIM,
                spatial=_COMPUTE_SPATIAL,
                precision_bits=precision_bits,
                quantize_activations=True,
            )
            for _ in range(_COMPUTE_PARALLEL_INSTANCES)
        )
        for kernel in compute_kernels:
            kernel.write_input(0, compute_input)

        stream_kernel = _compile_quantized_stream_kernel(
            channels=_STREAM_CHANNELS,
            spatial=_STREAM_SPATIAL,
        )
        stream_kernel.write_input(0, stream_input)
        baseline_kernels = tuple(
            _compile_quantized_static_conv_kernel(
                weights=compute_weights,
                input_channels=_COMPUTE_IN_DIM,
                output_channels=_COMPUTE_OUT_DIM,
                spatial=_COMPUTE_SPATIAL,
                precision_bits=precision_bits,
                quantize_activations=False,
            )
            for _ in range(_COMPUTE_PARALLEL_INSTANCES)
        )
        for kernel in baseline_kernels:
            kernel.write_input(0, compute_input)
        baseline_measurement = _measure_compute_tops(baseline_kernels)

    measurement = _measure_compute_tops(compute_kernels)
    activation_quantization_speedup = None
    native_quantized_compute = None
    if baseline_measurement is not None:
        activation_quantization_speedup = (
            measurement.tops / baseline_measurement.tops
            if baseline_measurement.tops > 0
            else 0.0
        )
        native_quantized_compute = (
            activation_quantization_speedup >= _NATIVE_QUANTIZATION_SPEEDUP_THRESHOLD
        )

    return AnePrecisionProfile(
        precision_bits=precision_bits,
        weight_bits=precision_bits,
        activation_bits=activation_bits,
        supported=True,
        compute_tops=measurement.tops,
        weight_only_compute_tops=(
            baseline_measurement.tops if baseline_measurement is not None else None
        ),
        single_instance_compute_tops=measurement.single_instance_tops,
        compute_instances=measurement.instances,
        memory_bandwidth_gbps=_measure_streaming_bandwidth_gbps(
            stream_kernel,
            bytes_per_element=(
                _FP32_BYTES_PER_ELEMENT
                if activation_bits == 32
                else _FP16_BYTES_PER_ELEMENT
            ),
        ),
        activation_quantization_speedup=activation_quantization_speedup,
        native_quantized_compute=native_quantized_compute,
    )


def _short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ")
    if len(message) <= 220:
        return message
    return f"{message[:217]}..."


@final
class _AneKernel:
    def __init__(
        self,
        *,
        model: Pointer,
        request: Pointer,
        input_surfaces: Sequence[Pointer],
        output_surfaces: Sequence[Pointer],
        tmp_dir: Path,
    ) -> None:
        self._model = model
        self._request = request
        self._input_surfaces = tuple(input_surfaces)
        self._output_surfaces = tuple(output_surfaces)
        self._tmp_dir = tmp_dir
        self._empty_dict = _ns_empty_dict()

    def __del__(self) -> None:
        try:
            err_ptr = ctypes.c_void_p(0)
            unload_fn = ctypes.cast(
                _require_objc_msg_send_address(),
                ctypes.CFUNCTYPE(
                    ctypes.c_bool,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_uint,
                    ctypes.POINTER(ctypes.c_void_p),
                ),
            )
            unload_fn(
                ctypes.c_void_p(self._model),
                ctypes.c_void_p(_sel("unloadWithQoS:error:")),
                _QOS_USER_INITIATED,
                ctypes.byref(err_ptr),
            )
        except Exception:
            pass

    def eval(self) -> None:
        err_ptr = ctypes.c_void_p(0)
        eval_fn = ctypes.cast(
            _require_objc_msg_send_address(),
            ctypes.CFUNCTYPE(
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ),
        )
        ok = cast(
            bool,
            eval_fn(
                ctypes.c_void_p(self._model),
                ctypes.c_void_p(_sel("evaluateWithQoS:options:request:error:")),
                _QOS_USER_INITIATED,
                ctypes.c_void_p(self._empty_dict),
                ctypes.c_void_p(self._request),
                ctypes.byref(err_ptr),
            ),
        )
        if not ok:
            err_desc = _describe_error(err_ptr.value)
            raise RuntimeError(f"ANE eval failed: {err_desc}")

    def write_input(self, index: int, values: NumericArray) -> None:
        _write_iosurface(self._input_surfaces[index], values)


def ane_available() -> bool:
    try:
        _load_ane_framework()
        return True
    except Exception:
        return False


def _measure_compute_tops(kernels: Sequence[_AneKernel]) -> _ComputeMeasurement:
    operations_per_iteration = 2 * _COMPUTE_IN_DIM * _COMPUTE_OUT_DIM * _COMPUTE_SPATIAL
    single_instance_tops = _measure_parallel_compute_tops(
        kernels[:1], operations_per_iteration=operations_per_iteration
    )
    best = _ComputeMeasurement(
        tops=single_instance_tops,
        single_instance_tops=single_instance_tops,
        instances=1,
    )
    if len(kernels) >= 2:
        parallel_tops = _measure_parallel_compute_tops(
            kernels[:2], operations_per_iteration=operations_per_iteration
        )
        if parallel_tops > best.tops:
            best = _ComputeMeasurement(
                tops=parallel_tops,
                single_instance_tops=single_instance_tops,
                instances=2,
            )
    return best


def _measure_parallel_compute_tops(
    kernels: Sequence[_AneKernel], *, operations_per_iteration: int
) -> float:
    if not kernels:
        return 0.0
    _warm_up_kernels(kernels, time.perf_counter() + _WARMUP_SECONDS)

    best_tops = 0.0
    with ThreadPoolExecutor(max_workers=len(kernels)) as executor:
        for _ in range(_MEASUREMENT_PASSES):
            start = time.perf_counter()
            if len(kernels) == 1:
                _eval_kernel_iterations(kernels[0], _COMPUTE_ITERATIONS_PER_PASS)
            else:
                futures = [
                    executor.submit(
                        _eval_kernel_iterations,
                        kernel,
                        _COMPUTE_ITERATIONS_PER_PASS,
                    )
                    for kernel in kernels
                ]
                for future in futures:
                    future.result()
            elapsed = time.perf_counter() - start
            if elapsed <= 0:
                continue
            tops = (
                operations_per_iteration
                * _COMPUTE_ITERATIONS_PER_PASS
                * len(kernels)
                / elapsed
                / 1e12
            )
            best_tops = max(best_tops, tops)
    return best_tops


def _warm_up_kernels(kernels: Sequence[_AneKernel], deadline: float) -> None:
    if len(kernels) == 1:
        _warm_up_with(kernels[0].eval, deadline)
        return
    with ThreadPoolExecutor(max_workers=len(kernels)) as executor:
        while time.perf_counter() < deadline:
            futures = [executor.submit(kernel.eval) for kernel in kernels]
            for future in futures:
                future.result()


def _eval_kernel_iterations(kernel: _AneKernel, iterations: int) -> None:
    for _ in range(iterations):
        kernel.eval()


def _measure_streaming_bandwidth_gbps(
    kernel: _AneKernel, *, bytes_per_element: int
) -> float:
    _warm_up_with(kernel.eval, time.perf_counter() + _WARMUP_SECONDS)

    bytes_per_iteration = _STREAM_CHANNELS * _STREAM_SPATIAL * bytes_per_element * 2
    best_gbps = 0.0
    for _ in range(_MEASUREMENT_PASSES):
        start = time.perf_counter()
        for _ in range(_STREAM_ITERATIONS_PER_PASS):
            kernel.eval()
        elapsed = time.perf_counter() - start
        if elapsed <= 0:
            continue
        gbps = bytes_per_iteration * _STREAM_ITERATIONS_PER_PASS / elapsed / 1e9
        best_gbps = max(best_gbps, gbps)
    return best_gbps


def _warm_up_with(do_op: Callable[[], None], deadline: float) -> None:
    while time.perf_counter() < deadline:
        do_op()


def _compile_static_conv_kernel(
    *,
    weights: NumericArray,
    input_channels: int,
    output_channels: int,
    spatial: int,
    precision_bits: Literal[32, 16],
) -> _AneKernel:
    element_type = _floating_mil_type(precision_bits)
    mil = _mil_static_conv(
        input_channels=input_channels,
        output_channels=output_channels,
        spatial=spatial,
        element_type=element_type,
    )
    weight_blob = _build_floating_weight_blob_bytes(weights, precision_bits)
    bytes_per_element = _bytes_per_precision(precision_bits)
    input_bytes = input_channels * spatial * bytes_per_element
    output_bytes = output_channels * spatial * bytes_per_element
    return _compile_kernel(
        mil_text=mil,
        weight_files={"weight.bin": weight_blob},
        input_sizes=(input_bytes,),
        output_sizes=(output_bytes,),
    )


def _compile_quantized_static_conv_kernel(
    *,
    weights: Float32Array,
    input_channels: int,
    output_channels: int,
    spatial: int,
    precision_bits: Literal[8, 4],
    quantize_activations: bool,
) -> _AneKernel:
    weight_blob = _build_quantized_weight_blob_bytes(weights, precision_bits)
    mil = _mil_quantized_static_conv(
        input_channels=input_channels,
        output_channels=output_channels,
        spatial=spatial,
        precision_bits=precision_bits,
        quantize_activations=quantize_activations,
    )
    input_bytes = input_channels * spatial * _FP16_BYTES_PER_ELEMENT
    output_bytes = output_channels * spatial * _FP16_BYTES_PER_ELEMENT
    return _compile_kernel(
        mil_text=mil,
        weight_files={"weight.bin": weight_blob},
        input_sizes=(input_bytes,),
        output_sizes=(output_bytes,),
    )


def _compile_relu_stream_kernel(
    *, channels: int, spatial: int, precision_bits: Literal[32, 16]
) -> _AneKernel:
    element_count = channels * spatial
    data_bytes = element_count * _bytes_per_precision(precision_bits)
    return _compile_kernel(
        mil_text=_mil_relu_stream(
            channels=channels,
            spatial=spatial,
            element_type=_floating_mil_type(precision_bits),
        ),
        weight_files={},
        input_sizes=(data_bytes,),
        output_sizes=(data_bytes,),
    )


def _compile_quantized_stream_kernel(*, channels: int, spatial: int) -> _AneKernel:
    element_count = channels * spatial
    data_bytes = element_count * _FP16_BYTES_PER_ELEMENT
    return _compile_kernel(
        mil_text=_mil_quantized_stream(channels=channels, spatial=spatial),
        weight_files={},
        input_sizes=(data_bytes,),
        output_sizes=(data_bytes,),
    )


def _floating_mil_type(precision_bits: Literal[32, 16]) -> Literal["fp32", "fp16"]:
    return "fp32" if precision_bits == 32 else "fp16"


def _bytes_per_precision(precision_bits: Literal[32, 16]) -> int:
    return _FP32_BYTES_PER_ELEMENT if precision_bits == 32 else _FP16_BYTES_PER_ELEMENT


def _compile_kernel(
    *,
    mil_text: str,
    weight_files: dict[str, bytes],
    input_sizes: Sequence[int],
    output_sizes: Sequence[int],
) -> _AneKernel:
    (
        descriptor_class,
        in_memory_model_class,
        request_class,
        io_surface_object_class,
    ) = _load_ane_framework()

    mil_bytes = mil_text.encode("utf-8")
    mil_data = _ns_data(mil_bytes)

    weight_keys: list[Pointer] = []
    weight_values: list[Pointer] = []
    for filename, raw_bytes in weight_files.items():
        entry = _ns_dict_from_keys_values(
            [_ns_str("offset"), _ns_str("data")],
            [_ns_int(0), _ns_data(raw_bytes)],
        )
        weight_keys.append(_ns_str(f"@model_path/weights/{filename}"))
        weight_values.append(entry)
    weights_dict = (
        _ns_dict_from_keys_values(weight_keys, weight_values)
        if weight_keys
        else _ns_empty_dict()
    )

    descriptor_fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ),
    )
    descriptor = _checked_pointer(
        cast(
            int | None,
            descriptor_fn(
                ctypes.c_void_p(descriptor_class),
                ctypes.c_void_p(_sel("modelWithMILText:weights:optionsPlist:")),
                ctypes.c_void_p(mil_data),
                ctypes.c_void_p(weights_dict),
                None,
            ),
        ),
        "ANE: modelWithMILText failed",
    )

    model = _id_id(
        in_memory_model_class,
        _sel("inMemoryModelWithDescriptor:"),
        descriptor,
        "ANE: inMemoryModelWithDescriptor failed",
    )

    tmp_dir = _write_model_files(model, mil_bytes, weight_files)

    _compile_and_load_model(model)

    input_surfaces = tuple(_create_iosurface(size) for size in input_sizes)
    output_surfaces = tuple(_create_iosurface(size) for size in output_sizes)
    for surface in (*input_surfaces, *output_surfaces):
        _zero_iosurface(surface)

    wrapped_inputs = _ns_mutable_array(len(input_surfaces))
    input_indices = _ns_mutable_array(len(input_surfaces))
    for index, surface in enumerate(input_surfaces):
        io_object = _id_id(
            io_surface_object_class,
            _sel("objectWithIOSurface:"),
            surface,
            "ANE: objectWithIOSurface failed for input",
        )
        _ns_array_add(wrapped_inputs, io_object)
        _ns_array_add(input_indices, _ns_int(index))

    wrapped_outputs = _ns_mutable_array(len(output_surfaces))
    output_indices = _ns_mutable_array(len(output_surfaces))
    for index, surface in enumerate(output_surfaces):
        io_object = _id_id(
            io_surface_object_class,
            _sel("objectWithIOSurface:"),
            surface,
            "ANE: objectWithIOSurface failed for output",
        )
        _ns_array_add(wrapped_outputs, io_object)
        _ns_array_add(output_indices, _ns_int(index))

    request_fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ),
    )
    request = _checked_pointer(
        cast(
            int | None,
            request_fn(
                ctypes.c_void_p(request_class),
                ctypes.c_void_p(
                    _sel(
                        "requestWithInputs:inputIndices:outputs:outputIndices:"
                        "weightsBuffer:perfStats:procedureIndex:"
                    )
                ),
                ctypes.c_void_p(wrapped_inputs),
                ctypes.c_void_p(input_indices),
                ctypes.c_void_p(wrapped_outputs),
                ctypes.c_void_p(output_indices),
                None,
                None,
                ctypes.c_void_p(_ns_int(0)),
            ),
        ),
        "ANE: requestWithInputs failed",
    )

    _retain(model)
    _retain(request)
    return _AneKernel(
        model=model,
        request=request,
        input_surfaces=input_surfaces,
        output_surfaces=output_surfaces,
        tmp_dir=tmp_dir,
    )


def _compile_and_load_model(model: Pointer) -> None:
    for selector, options in (
        ("compileWithQoS:options:error:", _ns_empty_dict()),
        ("loadWithQoS:options:error:", _load_options()),
    ):
        err_ptr = ctypes.c_void_p(0)
        compile_fn = ctypes.cast(
            _require_objc_msg_send_address(),
            ctypes.CFUNCTYPE(
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ),
        )
        ok = cast(
            bool,
            compile_fn(
                ctypes.c_void_p(model),
                ctypes.c_void_p(_sel(selector)),
                _QOS_USER_INITIATED,
                ctypes.c_void_p(options),
                ctypes.byref(err_ptr),
            ),
        )
        if not ok:
            err_desc = _describe_error(err_ptr.value)
            raise RuntimeError(f"ANE {selector} failed: {err_desc}")


def _load_options() -> Pointer:
    if not os.environ.get("EXO_ANE_KEEP_MODEL_WIRED"):
        return _ns_empty_dict()

    number_with_bool = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_bool,
        ),
    )
    yes = _checked_pointer(
        cast(
            int | None,
            number_with_bool(
                ctypes.c_void_p(_objc_class("NSNumber")),
                ctypes.c_void_p(_sel("numberWithBool:")),
                True,
            ),
        ),
        "NSNumber numberWithBool failed",
    )
    return _ns_dict_from_keys_values([_ns_str("kANEFKeepModelMemoryWiredKey")], [yes])


def _write_model_files(
    model: Pointer, mil_bytes: bytes, weight_files: dict[str, bytes]
) -> Path:
    hex_id = _id_none(model, _sel("hexStringIdentifier"))
    model_id = _to_cstr(hex_id) if _ns_string_length(hex_id) > 0 else str(uuid.uuid4())
    tmp_dir = Path(tempfile.gettempdir()) / model_id
    weights_dir = tmp_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "model.mil").write_bytes(mil_bytes)
    for filename, raw_bytes in weight_files.items():
        (weights_dir / filename).write_bytes(raw_bytes)
    return tmp_dir


def _write_iosurface(surface: Pointer, values: NumericArray) -> None:
    raw = cast(UInt8Array, np.ascontiguousarray(values).view(np.uint8).reshape(-1))
    nbytes = int(raw.nbytes)
    if nbytes > _iosurface_alloc_size(surface):
        raise ValueError("input data is larger than the IOSurface allocation")

    _iosurface_lock(surface, readonly=False)
    try:
        address = _iosurface_base_address(surface)
        ctypes.memmove(address, int(raw.ctypes.data), nbytes)
    finally:
        _iosurface_unlock(surface, readonly=False)


def _zero_iosurface(surface: Pointer) -> None:
    _iosurface_lock(surface, readonly=False)
    try:
        ctypes.memset(
            _iosurface_base_address(surface), 0, _iosurface_alloc_size(surface)
        )
    finally:
        _iosurface_unlock(surface, readonly=False)


def _create_iosurface(nbytes: int) -> Pointer:
    if nbytes <= 0:
        nbytes = 4
    keys = [
        _require_iosurface_key(_k_iosurface_width, "kIOSurfaceWidth"),
        _require_iosurface_key(_k_iosurface_height, "kIOSurfaceHeight"),
        _require_iosurface_key(
            _k_iosurface_bytes_per_element, "kIOSurfaceBytesPerElement"
        ),
        _require_iosurface_key(_k_iosurface_bytes_per_row, "kIOSurfaceBytesPerRow"),
        _require_iosurface_key(_k_iosurface_alloc_size, "kIOSurfaceAllocSize"),
        _require_iosurface_key(_k_iosurface_pixel_format, "kIOSurfacePixelFormat"),
    ]
    values = [
        _ns_ulong(nbytes),
        _ns_int(1),
        _ns_int(1),
        _ns_ulong(nbytes),
        _ns_ulong(nbytes),
        _ns_int(0),
    ]
    dictionary = _ns_dict_from_keys_values(keys, values)
    surface = _checked_pointer(
        cast(
            int | None,
            _require_iosurface().IOSurfaceCreate(ctypes.c_void_p(dictionary)),
        ),
        "IOSurfaceCreate failed",
    )
    return surface


def _iosurface_lock(surface: Pointer, *, readonly: bool) -> None:
    flags = 1 if readonly else 0
    result = cast(
        int,
        _require_iosurface().IOSurfaceLock(
            ctypes.c_void_p(surface), ctypes.c_uint32(flags), None
        ),
    )
    if result != 0:
        raise RuntimeError(f"IOSurfaceLock failed: {result}")


def _iosurface_unlock(surface: Pointer, *, readonly: bool) -> None:
    flags = 1 if readonly else 0
    result = cast(
        int,
        _require_iosurface().IOSurfaceUnlock(
            ctypes.c_void_p(surface), ctypes.c_uint32(flags), None
        ),
    )
    if result != 0:
        raise RuntimeError(f"IOSurfaceUnlock failed: {result}")


def _iosurface_base_address(surface: Pointer) -> Pointer:
    return _checked_pointer(
        cast(
            int | None,
            _require_iosurface().IOSurfaceGetBaseAddress(ctypes.c_void_p(surface)),
        ),
        "IOSurfaceGetBaseAddress failed",
    )


def _iosurface_alloc_size(surface: Pointer) -> int:
    return cast(
        int,
        _require_iosurface().IOSurfaceGetAllocSize(ctypes.c_void_p(surface)),
    )


def _build_floating_weight_blob_bytes(
    values: NumericArray, precision_bits: Literal[32, 16]
) -> bytes:
    dtype = np.float32 if precision_bits == 32 else np.float16
    type_code = 0x02 if precision_bits == 32 else 0x01
    flat = np.ascontiguousarray(values.reshape(-1), dtype=dtype)
    payload = flat.tobytes()
    buf = _build_multi_weight_blob_bytes([(type_code, payload)])
    return bytes(buf)


def _build_quantized_weight_blob_bytes(
    values: Float32Array, precision_bits: Literal[8, 4]
) -> bytes:
    weights = np.ascontiguousarray(values, dtype=np.float32)
    abs_weights = cast(Float32Array, np.abs(weights))
    channel_max = cast(Float32Array, np.max(abs_weights, axis=1, keepdims=True))
    max_abs = cast(Float32Array, np.maximum(channel_max, np.float32(1e-6)))
    if precision_bits == 8:
        scale = np.ascontiguousarray(
            (max_abs / np.float32(127.0)).reshape(-1, 1, 1, 1),
            dtype=np.float16,
        )
        quantized = np.clip(
            np.round(weights / max_abs * np.float32(127.0)), -128, 127
        ).astype(np.int8)
        data_payload = np.ascontiguousarray(quantized).tobytes()
        data_type_code = 0x04
    else:
        scale = np.ascontiguousarray(
            (max_abs / np.float32(7.0)).reshape(-1, 1, 1, 1),
            dtype=np.float16,
        )
        quantized = np.clip(
            np.round(weights / max_abs * np.float32(7.0)), -8, 7
        ).astype(np.int8)
        data_payload = _pack_int4_payload(quantized)
        data_type_code = 0x08

    return _build_multi_weight_blob_bytes(
        [
            (data_type_code, data_payload),
            (0x01, scale.tobytes()),
        ]
    )


def _pack_int4_payload(values: npt.NDArray[np.int8]) -> bytes:
    flat = np.ascontiguousarray(values.reshape(-1), dtype=np.int8)
    if flat.size % 2 != 0:
        flat = np.pad(flat, (0, 1)).astype(np.int8)
    nibbles = (flat & 0x0F).astype(np.uint8)
    packed = nibbles[0::2] | (nibbles[1::2] << 4)
    return np.ascontiguousarray(packed, dtype=np.uint8).tobytes()


def _build_multi_weight_blob_bytes(blobs: Sequence[tuple[int, bytes]]) -> bytes:
    total_bytes = 64
    descriptors_and_payloads: list[tuple[bytes, bytes]] = []
    for type_code, payload in blobs:
        descriptor = bytearray(64)
        _pack_uint32_le(descriptor, 0, 0xDEADBEEF)
        _pack_uint32_le(descriptor, 4, type_code)
        _pack_uint32_le(descriptor, 8, len(payload))
        _pack_uint32_le(descriptor, 16, total_bytes + 64)
        descriptors_and_payloads.append((bytes(descriptor), payload))
        total_bytes += 64 + len(payload)

    buf = bytearray(total_bytes)
    _pack_uint32_le(buf, 0, len(blobs))
    _pack_uint32_le(buf, 4, 0x02)

    offset = 64
    for descriptor, payload in descriptors_and_payloads:
        buf[offset : offset + 64] = descriptor
        offset += 64
        buf[offset : offset + len(payload)] = payload
        offset += len(payload)
    return bytes(buf)


def _pack_uint32_le(buf: bytearray, offset: int, value: int) -> None:
    buf[offset : offset + 4] = value.to_bytes(4, byteorder="little", signed=False)


def _load_ane_framework() -> tuple[Pointer, Pointer, Pointer, Pointer]:
    global _ane_loaded
    global _ane_descriptor_class
    global _ane_in_memory_model_class
    global _ane_request_class
    global _ane_io_surface_object_class

    if _ane_loaded:
        return (
            _require_pointer(_ane_descriptor_class, "_ANEInMemoryModelDescriptor"),
            _require_pointer(_ane_in_memory_model_class, "_ANEInMemoryModel"),
            _require_pointer(_ane_request_class, "_ANERequest"),
            _require_pointer(_ane_io_surface_object_class, "_ANEIOSurfaceObject"),
        )

    _load_objc_runtime()
    _load_iosurface_runtime()
    ctypes.cdll.LoadLibrary(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
        "AppleNeuralEngine"
    )

    _ane_descriptor_class = _objc_class("_ANEInMemoryModelDescriptor")
    _ane_in_memory_model_class = _objc_class("_ANEInMemoryModel")
    _ane_request_class = _objc_class("_ANERequest")
    _ane_io_surface_object_class = _objc_class("_ANEIOSurfaceObject")
    _ane_loaded = True
    return (
        _ane_descriptor_class,
        _ane_in_memory_model_class,
        _ane_request_class,
        _ane_io_surface_object_class,
    )


def _load_objc_runtime() -> None:
    global _objc
    global _objc_msg_send_address

    if _objc is not None and _objc_msg_send_address is not None:
        return
    library_name = ctypes.util.find_library("objc")
    if library_name is None:
        raise RuntimeError("objc runtime not found")

    objc: ctypes.CDLL = ctypes.cdll.LoadLibrary(library_name)
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    _objc = objc
    _objc_msg_send_address = _checked_pointer(
        ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value,
        "objc_msgSend not found",
    )


def _load_iosurface_runtime() -> None:
    global _iosurface
    global _k_iosurface_width
    global _k_iosurface_height
    global _k_iosurface_bytes_per_element
    global _k_iosurface_bytes_per_row
    global _k_iosurface_alloc_size
    global _k_iosurface_pixel_format

    if _iosurface is not None:
        return

    iosurface: ctypes.CDLL = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/IOSurface.framework/IOSurface"
    )
    iosurface.IOSurfaceCreate.restype = ctypes.c_void_p
    iosurface.IOSurfaceCreate.argtypes = [ctypes.c_void_p]
    iosurface.IOSurfaceLock.restype = ctypes.c_int
    iosurface.IOSurfaceLock.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    iosurface.IOSurfaceUnlock.restype = ctypes.c_int
    iosurface.IOSurfaceUnlock.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    iosurface.IOSurfaceGetBaseAddress.restype = ctypes.c_void_p
    iosurface.IOSurfaceGetBaseAddress.argtypes = [ctypes.c_void_p]
    iosurface.IOSurfaceGetAllocSize.restype = ctypes.c_size_t
    iosurface.IOSurfaceGetAllocSize.argtypes = [ctypes.c_void_p]

    _iosurface = iosurface
    _k_iosurface_width = _checked_pointer(
        ctypes.c_void_p.in_dll(iosurface, "kIOSurfaceWidth").value,
        "kIOSurfaceWidth not found",
    )
    _k_iosurface_height = _checked_pointer(
        ctypes.c_void_p.in_dll(iosurface, "kIOSurfaceHeight").value,
        "kIOSurfaceHeight not found",
    )
    _k_iosurface_bytes_per_element = _checked_pointer(
        ctypes.c_void_p.in_dll(iosurface, "kIOSurfaceBytesPerElement").value,
        "kIOSurfaceBytesPerElement not found",
    )
    _k_iosurface_bytes_per_row = _checked_pointer(
        ctypes.c_void_p.in_dll(iosurface, "kIOSurfaceBytesPerRow").value,
        "kIOSurfaceBytesPerRow not found",
    )
    _k_iosurface_alloc_size = _checked_pointer(
        ctypes.c_void_p.in_dll(iosurface, "kIOSurfaceAllocSize").value,
        "kIOSurfaceAllocSize not found",
    )
    _k_iosurface_pixel_format = _checked_pointer(
        ctypes.c_void_p.in_dll(iosurface, "kIOSurfacePixelFormat").value,
        "kIOSurfacePixelFormat not found",
    )


def _require_objc() -> ctypes.CDLL:
    if _objc is None:
        raise RuntimeError("objc runtime was not loaded")
    return _objc


def _require_iosurface() -> ctypes.CDLL:
    if _iosurface is None:
        raise RuntimeError("IOSurface runtime was not loaded")
    return _iosurface


def _require_objc_msg_send_address() -> Pointer:
    return _require_pointer(_objc_msg_send_address, "objc_msgSend")


def _require_iosurface_key(value: Pointer | None, name: str) -> Pointer:
    return _require_pointer(value, name)


def _require_pointer(value: Pointer | None, name: str) -> Pointer:
    if value is None or value == 0:
        raise RuntimeError(f"{name} is unavailable")
    return value


def _checked_pointer(value: object, context: str) -> Pointer:
    pointer = cast(int | None, value)
    if pointer is None or pointer == 0:
        raise RuntimeError(context)
    return pointer


def _objc_class(name: str) -> Pointer:
    return _checked_pointer(
        cast(int | None, _require_objc().objc_getClass(name.encode("utf-8"))),
        f"Objective-C class {name} not found",
    )


def _sel(name: str) -> Pointer:
    return _checked_pointer(
        cast(int | None, _require_objc().sel_registerName(name.encode("utf-8"))),
        f"Objective-C selector {name} not found",
    )


def _id_none(obj: Pointer, selector: Pointer) -> Pointer:
    fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
    )
    return _checked_pointer(
        cast(int | None, fn(ctypes.c_void_p(obj), ctypes.c_void_p(selector))),
        "Objective-C call failed",
    )


def _id_id(obj: Pointer, selector: Pointer, arg: Pointer, context: str) -> Pointer:
    fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ),
    )
    return _checked_pointer(
        cast(
            int | None,
            fn(ctypes.c_void_p(obj), ctypes.c_void_p(selector), ctypes.c_void_p(arg)),
        ),
        context,
    )


def _ns_str(value: str) -> Pointer:
    fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        ),
    )
    return _checked_pointer(
        cast(
            int | None,
            fn(
                ctypes.c_void_p(_objc_class("NSString")),
                ctypes.c_void_p(_sel("stringWithUTF8String:")),
                value.encode("utf-8"),
            ),
        ),
        "NSString stringWithUTF8String failed",
    )


def _ns_int(value: int) -> Pointer:
    fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ),
    )
    return _checked_pointer(
        cast(
            int | None,
            fn(
                ctypes.c_void_p(_objc_class("NSNumber")),
                ctypes.c_void_p(_sel("numberWithInt:")),
                value,
            ),
        ),
        "NSNumber numberWithInt failed",
    )


def _ns_ulong(value: int) -> Pointer:
    fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ),
    )
    return _checked_pointer(
        cast(
            int | None,
            fn(
                ctypes.c_void_p(_objc_class("NSNumber")),
                ctypes.c_void_p(_sel("numberWithUnsignedLong:")),
                value,
            ),
        ),
        "NSNumber numberWithUnsignedLong failed",
    )


def _ns_data(value: bytes) -> Pointer:
    fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_ulong,
        ),
    )
    return _checked_pointer(
        cast(
            int | None,
            fn(
                ctypes.c_void_p(_objc_class("NSData")),
                ctypes.c_void_p(_sel("dataWithBytes:length:")),
                value,
                len(value),
            ),
        ),
        "NSData dataWithBytes failed",
    )


def _ns_empty_dict() -> Pointer:
    return _id_none(_objc_class("NSDictionary"), _sel("dictionary"))


def _ns_mutable_array(capacity: int) -> Pointer:
    fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ),
    )
    return _checked_pointer(
        cast(
            int | None,
            fn(
                ctypes.c_void_p(_objc_class("NSMutableArray")),
                ctypes.c_void_p(_sel("arrayWithCapacity:")),
                capacity,
            ),
        ),
        "NSMutableArray arrayWithCapacity failed",
    )


def _ns_array_add(array: Pointer, obj: Pointer) -> None:
    fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ),
    )
    fn(
        ctypes.c_void_p(array),
        ctypes.c_void_p(_sel("addObject:")),
        ctypes.c_void_p(obj),
    )


def _ns_dict_from_keys_values(
    keys: Sequence[Pointer], values: Sequence[Pointer]
) -> Pointer:
    if len(keys) != len(values):
        raise ValueError("keys and values must have the same length")
    if not keys:
        return _ns_empty_dict()

    count = len(keys)
    key_array_type = ctypes.c_void_p * count
    value_array_type = ctypes.c_void_p * count
    fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_ulong,
        ),
    )
    return _checked_pointer(
        cast(
            int | None,
            fn(
                ctypes.c_void_p(_objc_class("NSDictionary")),
                ctypes.c_void_p(_sel("dictionaryWithObjects:forKeys:count:")),
                value_array_type(*values),
                key_array_type(*keys),
                count,
            ),
        ),
        "NSDictionary dictionaryWithObjects failed",
    )


def _retain(obj: Pointer) -> None:
    _id_none(obj, _sel("retain"))


def _to_cstr(ns_string: Pointer) -> str:
    fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p),
    )
    raw = cast(
        bytes | None,
        fn(ctypes.c_void_p(ns_string), ctypes.c_void_p(_sel("UTF8String"))),
    )
    if raw is None:
        return ""
    return raw.decode("utf-8", errors="replace")


def _ns_string_length(ns_string: Pointer) -> int:
    fn = ctypes.cast(
        _require_objc_msg_send_address(),
        ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p),
    )
    return cast(int, fn(ctypes.c_void_p(ns_string), ctypes.c_void_p(_sel("length"))))


def _describe_error(error_pointer: int | None) -> str:
    if error_pointer is None or error_pointer == 0:
        return ""

    description = _id_none(error_pointer, _sel("description"))
    return _to_cstr(description)


_MIL_HEADER = (
    "program(1.3)\n"
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n'
    "{\n"
)

_CONV_PARAMS = (
    '        string pt = const()[name=string("pt"), val=string("valid")];\n'
    '        tensor<int32, [2]> st = const()[name=string("st"), '
    "val=tensor<int32, [2]>([1,1])];\n"
    '        tensor<int32, [4]> pd = const()[name=string("pd"), '
    "val=tensor<int32, [4]>([0,0,0,0])];\n"
    '        tensor<int32, [2]> dl = const()[name=string("dl"), '
    "val=tensor<int32, [2]>([1,1])];\n"
    '        int32 gr = const()[name=string("gr"), val=int32(1)];\n'
)

_CONV_ARGS = "dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st"


def _mil_static_conv(
    *,
    input_channels: int,
    output_channels: int,
    spatial: int,
    element_type: Literal["fp32", "fp16"],
) -> str:
    return (
        f"{_MIL_HEADER}"
        f"    func main<ios18>(tensor<{element_type}, [1, {input_channels}, 1, {spatial}]> x) {{\n"
        f"        tensor<{element_type}, [{output_channels}, {input_channels}, 1, 1]> W = const()"
        f'[name=string("W"), val=tensor<{element_type}, [{output_channels}, {input_channels}, 1, 1]>'
        '(BLOBFILE(path=string("@model_path/weights/weight.bin"), '
        "offset=uint64(64)))];\n"
        f"{_CONV_PARAMS}"
        f"        tensor<{element_type}, [1, {output_channels}, 1, {spatial}]> y = conv("
        f'{_CONV_ARGS}, weight=W, x=x)[name=string("cv")];\n'
        "    } -> (y);\n"
        "}\n"
    )


def _mil_quantized_static_conv(
    *,
    input_channels: int,
    output_channels: int,
    spatial: int,
    precision_bits: Literal[8, 4],
    quantize_activations: bool,
) -> str:
    data_type = "int8" if precision_bits == 8 else "int4"
    scale_blob_offset = 128 + _quantized_data_payload_size(
        input_channels=input_channels,
        output_channels=output_channels,
        precision_bits=precision_bits,
    )
    activation_preamble = (
        _mil_quantize_dequantize(channels=input_channels, spatial=spatial)
        if quantize_activations
        else ""
    )
    conv_input = "dx" if quantize_activations else "x"
    return (
        f"{_MIL_HEADER}"
        f"    func main<ios18>(tensor<fp16, [1, {input_channels}, 1, {spatial}]> x) {{\n"
        f"        tensor<fp16, [{output_channels}, {input_channels}, 1, 1]> W = "
        f"constexpr_blockwise_shift_scale(data = tensor<{data_type}, "
        f"[{output_channels}, {input_channels}, 1, 1]>(BLOBFILE(path = "
        'string("@model_path/weights/weight.bin"), offset = uint64(64))), '
        f"scale = tensor<fp16, [{output_channels}, 1, 1, 1]>(BLOBFILE(path = "
        'string("@model_path/weights/weight.bin"), '
        f'offset = uint64({scale_blob_offset}))))[name = string("Wq")];\n'
        f"{activation_preamble}"
        f"{_CONV_PARAMS}"
        f"        tensor<fp16, [1, {output_channels}, 1, {spatial}]> y = conv("
        f'{_CONV_ARGS}, weight=W, x={conv_input})[name=string("cv")];\n'
        "    } -> (y);\n"
        "}\n"
    )


def _mil_relu_stream(
    *, channels: int, spatial: int, element_type: Literal["fp32", "fp16"]
) -> str:
    return (
        f"{_MIL_HEADER}"
        f"    func main<ios18>(tensor<{element_type}, [1, {channels}, 1, {spatial}]> x) {{\n"
        f"        tensor<{element_type}, [1, {channels}, 1, {spatial}]> y = relu(x=x)"
        '[name=string("relu")];\n'
        "    } -> (y);\n"
        "}\n"
    )


def _mil_quantized_stream(*, channels: int, spatial: int) -> str:
    return (
        f"{_MIL_HEADER}"
        f"    func main<ios18>(tensor<fp16, [1, {channels}, 1, {spatial}]> x) {{\n"
        f"{_mil_quantize_dequantize(channels=channels, spatial=spatial)}"
        f"        tensor<fp16, [1, {channels}, 1, {spatial}]> y = relu(x=dx)"
        '[name=string("relu")];\n'
        "    } -> (y);\n"
        "}\n"
    )


def _quantized_data_payload_size(
    *, input_channels: int, output_channels: int, precision_bits: Literal[8, 4]
) -> int:
    element_count = input_channels * output_channels
    if precision_bits == 8:
        return element_count
    return (element_count + 1) // 2


def _mil_quantize_dequantize(*, channels: int, spatial: int) -> str:
    return (
        '        fp16 qs = const()[name = string("qs"), val = fp16(0x1p-7)];\n'
        '        int8 qz = const()[name = string("qz"), val = int8(0)];\n'
        '        string qdt = const()[name = string("qdt"), val = string("int8")];\n'
        f"        tensor<int8, [1, {channels}, 1, {spatial}]> qx = quantize("
        "input = x, output_dtype = qdt, scale = qs, zero_point = qz)"
        '[name = string("qx")];\n'
        '        fp16 dqs = const()[name = string("dqs"), val = fp16(0x1p-7)];\n'
        '        int8 dqz = const()[name = string("dqz"), val = int8(0)];\n'
        f"        tensor<fp16, [1, {channels}, 1, {spatial}]> dx = dequantize("
        'input = qx, scale = dqs, zero_point = dqz)[name = string("dx")];\n'
    )
