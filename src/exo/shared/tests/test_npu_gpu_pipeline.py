import pytest

from exo.shared.types.npu_gpu_pipeline import (
    HeterogeneousPipelineTelemetry,
    LayerAssignmentRule,
)
from exo.worker.engines.npu import (
    HeterogeneousPipelinedScheduler,
    PipelinedInferenceCoordinator,
)


def test_layer_assignment_rule() -> None:
    rule = LayerAssignmentRule(layer_index=0, prefill_device="gpu", decode_device="npu")
    assert rule.device_for_phase("prefill") == "gpu"
    assert rule.device_for_phase("decode") == "npu"


def test_pipeline_telemetry_properties() -> None:
    telemetry = HeterogeneousPipelineTelemetry(
        pipeline_id="test-pipeline",
        phase="decode",
        layers_executed_gpu=2,
        layers_executed_npu=6,
        gpu_execution_time_ms=10.0,
        npu_execution_time_ms=20.0,
        pipeline_overlap_time_ms=5.0,
        estimated_power_reduction_ratio=0.40,
    )
    assert telemetry.total_wall_time_ms == 25.0
    assert telemetry.overlap_ratio == 0.5  # 5.0 / min(10.0, 20.0)


def test_scheduler_phase_split_mode() -> None:
    scheduler = HeterogeneousPipelinedScheduler(total_layers=8, mode="phase_split", npu_available=True)
    assert scheduler.total_layers == 8

    rules = scheduler.get_layer_rules()
    assert len(rules) == 8

    for i in range(8):
        assert scheduler.target_device_for_layer(i, "prefill") == "gpu"
        assert scheduler.target_device_for_layer(i, "decode") == "npu"


def test_scheduler_layer_ping_pong_mode() -> None:
    scheduler = HeterogeneousPipelinedScheduler(total_layers=6, mode="layer_ping_pong", npu_available=True)

    # In prefill: even on GPU, odd on NPU
    assert scheduler.target_device_for_layer(0, "prefill") == "gpu"
    assert scheduler.target_device_for_layer(1, "prefill") == "npu"
    assert scheduler.target_device_for_layer(2, "prefill") == "gpu"
    assert scheduler.target_device_for_layer(3, "prefill") == "npu"

    # In decode: even on NPU, odd on GPU
    assert scheduler.target_device_for_layer(0, "decode") == "npu"
    assert scheduler.target_device_for_layer(1, "decode") == "gpu"


def test_scheduler_fallback_when_npu_unavailable() -> None:
    scheduler = HeterogeneousPipelinedScheduler(total_layers=4, mode="phase_split", npu_available=False)

    for i in range(4):
        assert scheduler.target_device_for_layer(i, "prefill") == "gpu"
        assert scheduler.target_device_for_layer(i, "decode") == "gpu"


@pytest.mark.asyncio
async def test_coordinator_prefill_and_decode_execution() -> None:
    coordinator = PipelinedInferenceCoordinator(total_layers=4, mode="phase_split", npu_available=True)

    def dummy_gpu_kernel(act: int, layer_idx: int) -> int:
        return act + 10

    def dummy_npu_kernel(act: int, layer_idx: int) -> int:
        return act + 1

    # 1. Test Prefill phase (all 4 layers on GPU)
    out_prefill, tel_prefill = await coordinator.execute_step(
        activations=0,
        phase="prefill",
        gpu_kernel=dummy_gpu_kernel,
        npu_kernel=dummy_npu_kernel,
    )
    # 0 + 10 * 4 = 40
    assert out_prefill == 40
    assert tel_prefill.layers_executed_gpu == 4
    assert tel_prefill.layers_executed_npu == 0
    assert tel_prefill.phase == "prefill"

    # 2. Test Decode phase (all 4 layers on NPU)
    out_decode, tel_decode = await coordinator.execute_step(
        activations=0,
        phase="decode",
        gpu_kernel=dummy_gpu_kernel,
        npu_kernel=dummy_npu_kernel,
    )
    # 0 + 1 * 4 = 4
    assert out_decode == 4
    assert tel_decode.layers_executed_gpu == 0
    assert tel_decode.layers_executed_npu == 4
    assert tel_decode.phase == "decode"
    assert tel_decode.estimated_power_reduction_ratio == 0.40


@pytest.mark.asyncio
async def test_coordinator_ping_pong_execution() -> None:
    coordinator = PipelinedInferenceCoordinator(total_layers=4, mode="layer_ping_pong", npu_available=True)

    def dummy_gpu_kernel(act: int, layer_idx: int) -> int:
        return act + 10

    def dummy_npu_kernel(act: int, layer_idx: int) -> int:
        return act + 1

    out, tel = await coordinator.execute_step(
        activations=0,
        phase="prefill",
        gpu_kernel=dummy_gpu_kernel,
        npu_kernel=dummy_npu_kernel,
    )

    # 2 layers on GPU (+10 * 2), 2 layers on NPU (+1 * 2) = 22
    assert out == 22
    assert tel.layers_executed_gpu == 2
    assert tel.layers_executed_npu == 2
