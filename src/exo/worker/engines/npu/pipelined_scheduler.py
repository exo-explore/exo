import asyncio
from collections.abc import Sequence

from exo.shared.types.npu_gpu_pipeline import (
    InferencePhase,
    LayerAssignmentRule,
    PipeliningMode,
    TargetDevice,
)


class HeterogeneousPipelinedScheduler:
    """Schedules model transformer layers across GPU and NPU compute engines."""

    def __init__(
        self,
        total_layers: int,
        mode: PipeliningMode = "phase_split",
        npu_available: bool = True,
    ) -> None:
        self._total_layers = total_layers
        self._mode = mode
        self._npu_available = npu_available
        self._layer_rules: list[LayerAssignmentRule] = []
        self._build_schedule()

    @property
    def total_layers(self) -> int:
        return self._total_layers

    @property
    def mode(self) -> PipeliningMode:
        return self._mode

    @property
    def npu_available(self) -> bool:
        return self._npu_available

    def _build_schedule(self) -> None:
        """Constructs layer assignment rules based on pipelining mode and device availability."""
        rules: list[LayerAssignmentRule] = []
        for i in range(self._total_layers):
            if not self._npu_available:
                # Fallback to GPU for all layers if NPU is absent
                rules.append(LayerAssignmentRule(layer_index=i, prefill_device="gpu", decode_device="gpu"))
            elif self._mode == "phase_split":
                # Classical Phase-Split: Prefill on GPU (compute-bound), Decode on NPU (memory-bound)
                rules.append(LayerAssignmentRule(layer_index=i, prefill_device="gpu", decode_device="npu"))
            elif self._mode == "layer_ping_pong":
                # Layer Ping-Pong: Even layers on GPU, odd layers on NPU for continuous pipelined concurrency
                prefill_dev: TargetDevice = "gpu" if i % 2 == 0 else "npu"
                decode_dev: TargetDevice = "npu" if i % 2 == 0 else "gpu"
                rules.append(LayerAssignmentRule(layer_index=i, prefill_device=prefill_dev, decode_device=decode_dev))
            else:
                # Default adaptive
                rules.append(LayerAssignmentRule(layer_index=i, prefill_device="gpu", decode_device="npu"))
        self._layer_rules = rules

    def get_layer_rules(self) -> Sequence[LayerAssignmentRule]:
        return tuple(self._layer_rules)

    def target_device_for_layer(self, layer_index: int, phase: InferencePhase) -> TargetDevice:
        """Determines the assigned device for a given layer index and inference phase."""
        if 0 <= layer_index < len(self._layer_rules):
            return self._layer_rules[layer_index].device_for_phase(phase)
        return "gpu"

    async def execute_layer_pipeline(
        self,
        activations: object,
        phase: InferencePhase,
        gpu_layer_kernel,
        npu_layer_kernel,
    ) -> tuple[object, int, int]:
        """Executes activations through the layer pipeline with ping-pong buffering.

        Returns:
            Tuple of (output_activations, layers_on_gpu, layers_on_npu).
        """
        curr = activations
        gpu_count = 0
        npu_count = 0

        for layer_idx in range(self._total_layers):
            target = self.target_device_for_layer(layer_idx, phase)
            if target == "gpu":
                curr = await asyncio.to_thread(gpu_layer_kernel, curr, layer_idx)
                gpu_count += 1
            else:
                curr = await asyncio.to_thread(npu_layer_kernel, curr, layer_idx)
                npu_count += 1

        return curr, gpu_count, npu_count
