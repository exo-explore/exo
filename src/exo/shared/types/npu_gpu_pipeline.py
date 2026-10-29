from typing import Literal

from exo.utils.pydantic_ext import FrozenModel

InferencePhase = Literal["prefill", "decode"]
TargetDevice = Literal["gpu", "npu", "cpu"]
PipeliningMode = Literal["phase_split", "layer_ping_pong", "adaptive"]


class LayerAssignmentRule(FrozenModel):
    """Specification defining which processor executes a specific model layer."""

    layer_index: int
    prefill_device: TargetDevice = "gpu"
    decode_device: TargetDevice = "npu"

    def device_for_phase(self, phase: InferencePhase) -> TargetDevice:
        return self.prefill_device if phase == "prefill" else self.decode_device


class HeterogeneousPipelineTelemetry(FrozenModel):
    """Telemetry capturing execution latency, layer distribution, and power savings."""

    pipeline_id: str
    phase: InferencePhase
    layers_executed_gpu: int
    layers_executed_npu: int
    gpu_execution_time_ms: float
    npu_execution_time_ms: float
    pipeline_overlap_time_ms: float = 0.0
    estimated_power_reduction_ratio: float = 0.40

    @property
    def total_wall_time_ms(self) -> float:
        return self.gpu_execution_time_ms + self.npu_execution_time_ms - self.pipeline_overlap_time_ms

    @property
    def overlap_ratio(self) -> float:
        min_time = min(self.gpu_execution_time_ms, self.npu_execution_time_ms)
        if min_time <= 0:
            return 0.0
        return min(1.0, self.pipeline_overlap_time_ms / min_time)
