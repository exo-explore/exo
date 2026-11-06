import time
import uuid

from exo.shared.types.npu_gpu_pipeline import (
    HeterogeneousPipelineTelemetry,
    InferencePhase,
    PipeliningMode,
)
from exo.worker.engines.npu.pipelined_scheduler import HeterogeneousPipelinedScheduler


class PipelinedInferenceCoordinator:
    """Coordinates heterogeneous multi-layer inference across GPU and NPU compute engines."""

    def __init__(
        self,
        total_layers: int = 32,
        mode: PipeliningMode = "phase_split",
        npu_available: bool = True,
    ) -> None:
        self._scheduler = HeterogeneousPipelinedScheduler(
            total_layers=total_layers,
            mode=mode,
            npu_available=npu_available,
        )
        self._step_counter: int = 0

    @property
    def scheduler(self) -> HeterogeneousPipelinedScheduler:
        return self._scheduler

    async def execute_step(
        self,
        activations: object,
        phase: InferencePhase,
        gpu_kernel,
        npu_kernel,
    ) -> tuple[object, HeterogeneousPipelineTelemetry]:
        """Executes a complete inference step across the scheduled layers and collects telemetry."""
        self._step_counter += 1
        pipeline_id = f"pipeline-step-{uuid.uuid4().hex[:8]}-{self._step_counter}"

        start_time = time.perf_counter()

        # Track execution timings for GPU and NPU layers
        gpu_times: list[float] = []
        npu_times: list[float] = []

        def timed_gpu_kernel(act: object, layer_idx: int) -> object:
            t0 = time.perf_counter()
            res = gpu_kernel(act, layer_idx)
            gpu_times.append((time.perf_counter() - t0) * 1000.0)
            return res

        def timed_npu_kernel(act: object, layer_idx: int) -> object:
            t0 = time.perf_counter()
            res = npu_kernel(act, layer_idx)
            npu_times.append((time.perf_counter() - t0) * 1000.0)
            return res

        result, gpu_layers, npu_layers = await self._scheduler.execute_layer_pipeline(
            activations=activations,
            phase=phase,
            gpu_layer_kernel=timed_gpu_kernel,
            npu_layer_kernel=timed_npu_kernel,
        )

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        gpu_total_ms = sum(gpu_times)
        npu_total_ms = sum(npu_times)

        # Estimate overlapping benefit in ping-pong or pipelined mode
        overlap_ms = max(0.0, (gpu_total_ms + npu_total_ms) - total_elapsed_ms) if (gpu_layers > 0 and npu_layers > 0) else 0.0

        # Estimate power reduction: NPU consumes ~20% of GPU power for decode
        power_reduction = 0.40 if npu_layers > gpu_layers else (0.20 if npu_layers > 0 else 0.0)

        telemetry = HeterogeneousPipelineTelemetry(
            pipeline_id=pipeline_id,
            phase=phase,
            layers_executed_gpu=gpu_layers,
            layers_executed_npu=npu_layers,
            gpu_execution_time_ms=gpu_total_ms,
            npu_execution_time_ms=npu_total_ms,
            pipeline_overlap_time_ms=overlap_ms,
            estimated_power_reduction_ratio=power_reduction,
        )

        return result, telemetry
