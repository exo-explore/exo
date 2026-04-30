"""GPU compute and memory bandwidth probe.

Runs a small dense FP16 matmul to estimate TFLOPS and a large `mx.sum` to
estimate memory bandwidth. Both are measured in a worker thread (MLX `eval`
is blocking) so the event loop is unaffected.

The probe is "active" — it loads the GPU. The caller (ProfilerManager) is
responsible for not running it while inference is active.

## Design

- **Why `mx.sum` instead of GEMV for bandwidth.** GEMV pays for an inner
  reduction plus a vector broadcast and caps at ~58% of peak DRAM bandwidth
  on Apple Silicon. A pure streaming reduction (`mx.sum`) over a multi-GB
  buffer hits ~80% of published peak — closer to the ceiling that's useful
  for placement decisions. M3 Ultra publishes 819 GB/s; a 2 GB `mx.sum`
  reports ~650 GB/s when the GPU is in its peak performance state.

- **The buffer must defeat the SLC.** M3 Ultra has ~96 MB system-level
  cache. 2 GB sits well past the cache plateau.

- **Variance control.** Apple Silicon GPUs aggressively scale clocks based
  on demand. A short cold benchmark catches the GPU mid-ramp-up and
  reports a number 15-20% lower than the hardware can do. Two M3 Ultras
  measured cold can differ by 100 GB/s on the *same* chip-and-board pair
  even though peak silicon performance is identical.

  The fix is what every hardware reviewer does:
    1. **Long warm-up** — drive the workload for ~1 second to let macOS
       lock the GPU into its peak performance state.
    2. **Best-of-N** — time several short measurement passes and report
       the *fastest*. The best run represents what the silicon can do
       when it isn't being interrupted by Spotlight, the WindowServer,
       or thermal pressure. Slower runs are noise from the rest of the
       system, not a property of the GPU.

  This makes two healthy M3 Ultras agree to within a few percent.
"""

import time
from collections.abc import Callable
from typing import Literal, Self, final

import mlx.core as mx
from anyio import to_thread

from exo.utils.pydantic_ext import TaggedModel

_MATMUL_DIM = 4096
_MATMUL_ITERATIONS_PER_PASS = 8
_BANDWIDTH_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB streaming buffer
_BANDWIDTH_ITERATIONS_PER_PASS = 16
_DTYPE = mx.float16
_BYTES_PER_ELEMENT = 2

# Drive the workload until the GPU is in its peak performance state. macOS
# typically clocks up within a few hundred ms of sustained load; 1.0 s is
# generous.
_WARMUP_SECONDS = 1.0
# Number of timed passes; we report the best one.
_MEASUREMENT_PASSES = 5


@final
class GpuProfile(TaggedModel):
    """Wire format for a measured GPU profile, gathered locally on a node."""

    engine: Literal["mlx"]
    tflops_fp16: float
    memory_bandwidth_gbps: float

    @classmethod
    async def measure(cls) -> Self | None:
        if not mx.metal.is_available():
            return None
        return await to_thread.run_sync(cls._measure_blocking)

    @classmethod
    def _measure_blocking(cls) -> Self:
        return cls(
            engine="mlx",
            tflops_fp16=_measure_matmul_tflops(),
            memory_bandwidth_gbps=_measure_streaming_bandwidth_gbps(),
        )


def _warm_up_with(do_op: Callable[[], mx.array], deadline: float) -> None:
    """Repeatedly run `do_op` and `mx.eval` it until `deadline` (perf_counter
    seconds), so that the GPU reaches its peak performance state before any
    timed iteration begins. We discard the return value.
    """
    while time.perf_counter() < deadline:
        mx.eval(do_op())


def _measure_matmul_tflops() -> float:
    """Time a square FP16 matmul and return effective TFLOPS.

    `mx.eval` is called inside the loop because MLX's lazy graph would
    otherwise fuse all iterations into a single matmul on the trailing eval —
    we'd then divide one matmul's runtime by N and overestimate TFLOPS.
    """
    a = mx.random.uniform(shape=(_MATMUL_DIM, _MATMUL_DIM), dtype=_DTYPE)
    b = mx.random.uniform(shape=(_MATMUL_DIM, _MATMUL_DIM), dtype=_DTYPE)
    mx.eval(a, b)

    _warm_up_with(lambda: mx.matmul(a, b), time.perf_counter() + _WARMUP_SECONDS)

    flops_per_iteration = 2 * _MATMUL_DIM * _MATMUL_DIM * _MATMUL_DIM
    best_tflops = 0.0
    for _ in range(_MEASUREMENT_PASSES):
        start = time.perf_counter()
        for _ in range(_MATMUL_ITERATIONS_PER_PASS):
            mx.eval(mx.matmul(a, b))
        elapsed = time.perf_counter() - start
        if elapsed <= 0:
            continue
        total_flops = flops_per_iteration * _MATMUL_ITERATIONS_PER_PASS
        tflops = total_flops / elapsed / 1e12
        if tflops > best_tflops:
            best_tflops = tflops
    return best_tflops


def _measure_streaming_bandwidth_gbps() -> float:
    """Time a pure-read reduction and infer memory bandwidth from bytes streamed.

    `mx.sum` over a multi-GB buffer is a near-pure memory read with a tiny
    add per element — the bandwidth-bound code path that comes closest to
    the chip's DRAM ceiling on Apple Silicon. Same eval-per-iteration
    discipline as the matmul: without it MLX fuses the loop into one
    reduction and we'd report a wildly optimistic number.
    """
    n_elements = _BANDWIDTH_BYTES // _BYTES_PER_ELEMENT
    buffer = mx.random.uniform(shape=(n_elements,), dtype=_DTYPE)
    mx.eval(buffer)

    _warm_up_with(lambda: mx.sum(buffer), time.perf_counter() + _WARMUP_SECONDS)

    best_gbps = 0.0
    for _ in range(_MEASUREMENT_PASSES):
        start = time.perf_counter()
        for _ in range(_BANDWIDTH_ITERATIONS_PER_PASS):
            mx.eval(mx.sum(buffer))
        elapsed = time.perf_counter() - start
        if elapsed <= 0:
            continue
        total_bytes = _BANDWIDTH_BYTES * _BANDWIDTH_ITERATIONS_PER_PASS
        gbps = total_bytes / elapsed / 1e9
        if gbps > best_gbps:
            best_gbps = gbps
    return best_gbps
