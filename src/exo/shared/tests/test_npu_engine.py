from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from exo.shared.models.model_cards import Backend, ModelCard, ModelTask
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, MlxRingInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.info_gatherer.info_gatherer import (
    NodeBackends,
    _has_apple_ane,
    _has_pc_npu,
)
from exo.worker.engines.npu.builder import NpuBuilder
from exo.worker.engines.npu.device import NpuArchitecture, NpuDeviceProfile
from exo.worker.engines.npu.engine import NpuEngine
from exo.worker.engines.npu.runtime import NpuExecutionSession


def test_npu_backend_enums_and_values() -> None:
    """Verify Backend enum contains AppleAne and PcNpu with correct string values."""
    assert Backend.AppleAne == "AppleAne"
    assert Backend.PcNpu == "PcNpu"
    assert Backend("AppleAne") is Backend.AppleAne
    assert Backend("PcNpu") is Backend.PcNpu


def test_hardware_probing_apple_ane_and_pc_npu() -> None:
    """Verify hardware probing logic for Apple Silicon ANE and PC NPU."""
    # Under Darwin arm64, ANE should be detected
    with (
        patch("exo.utils.info_gatherer.info_gatherer.IS_DARWIN", True),
        patch("platform.machine", return_value="arm64"),
    ):
        assert _has_apple_ane() is True

    # Under non-Darwin, ANE should return False
    with patch("exo.utils.info_gatherer.info_gatherer.IS_DARWIN", False):
        assert _has_apple_ane() is False

    # Under EXO_ENABLE_NPU=1, PC NPU probe returns True
    with patch.dict(os.environ, {"EXO_ENABLE_NPU": "1"}):
        assert _has_pc_npu() is True


@pytest.mark.asyncio
async def test_node_backends_gather_includes_npu() -> None:
    """Verify NodeBackends.gather collects AppleAne and PcNpu when available."""
    with (
        patch("exo.utils.info_gatherer.info_gatherer._has_apple_ane", return_value=True),
        patch("exo.utils.info_gatherer.info_gatherer._has_pc_npu", return_value=True),
        patch("exo.utils.info_gatherer.info_gatherer._has_nvml_cuda", return_value=False),
    ):
        gathered = await NodeBackends.gather()
        assert Backend.AppleAne in gathered.backends
        assert Backend.PcNpu in gathered.backends


def test_npu_device_profile_and_energy_estimation() -> None:
    """Verify NpuDeviceProfile creation, validation, and energy estimation."""
    profile = NpuDeviceProfile(
        architecture=NpuArchitecture.AppleANE,
        tera_operations_per_second=15.8,
        memory_bandwidth_gigabytes=100.0,
        thermal_efficiency_factor=4.5,
        supports_float16=True,
        supports_int8=True,
        supports_dynamic_shapes=False,
    )
    assert profile.architecture == NpuArchitecture.AppleANE
    assert profile.thermal_efficiency_factor == 4.5

    # 100 tokens with 7B parameter model
    joules = profile.estimate_energy_joules(token_count=100, parameter_count_billions=7.0)
    assert joules > 0.0
    # Higher efficiency should result in significantly lower energy consumption
    baseline_gpu_joules = 0.05 * 100 * 7.0
    assert np.isclose(joules, baseline_gpu_joules / 4.5)


def test_npu_execution_session_lifecycle_and_compatibility() -> None:
    """Verify NpuExecutionSession graph initialization, forward, and operation filtering."""
    profile = NpuDeviceProfile.detect_local_device()
    session = NpuExecutionSession(
        device_profile=profile,
        model_path="test-model",
        hidden_dimension=128,
    )
    assert not session.is_initialized

    session.initialize_graph()
    assert session.is_initialized

    # Test compatibility checks
    assert session.is_operation_npu_compatible("matmul") is True
    assert session.is_operation_npu_compatible("dynamic_rope") is False

    # Test forward projection
    test_input = np.ones((1, 128), dtype=np.float32)
    weights = np.eye(128, dtype=np.float32)
    output = session.execute_layer_forward(test_input, layer_weights=weights)
    assert np.allclose(output, test_input)
    assert session.total_tokens_executed == 1

    session.close()
    assert not session.is_initialized


def test_npu_builder_and_engine_integration() -> None:
    """Verify NpuBuilder creates and executes NpuEngine through the full lifecycle."""
    model_id = ModelId("test-npu-model")
    model_card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(500),
        n_layers=3,
        hidden_size=256,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.AppleAne],
    )

    runner_id = RunnerId("runner-npu-1")
    node_id = NodeId("node-npu-1")

    shard_metadata = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=3,
        n_layers=3,
    )

    instance = MlxRingInstance(
        instance_id="inst-1",
        shard_assignments=ShardAssignments(
            model_id=model_id,
            runner_to_shard={runner_id: shard_metadata},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={node_id: []},
        ephemeral_port=8080,
    )

    bound_instance = BoundInstance(
        instance=instance,
        bound_runner_id=runner_id,
        bound_node_id=node_id,
    )

    builder = NpuBuilder(model_id=model_id)
    builder.connect(bound_instance)
    assert builder.device_profile is not None

    # Load generator
    loading_responses = list(builder.load(bound_instance))
    assert len(loading_responses) == 3
    assert loading_responses[-1].layers_loaded == 3

    # Build engine
    engine = builder.build()
    assert isinstance(engine, NpuEngine)

    engine.warmup()

    # Submit task and step
    task_id = TaskId("task-1")
    task_params = TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content=InputMessageContent("Hello NPU"))],
        max_output_tokens=2,
    )
    task = TextGeneration(
        task_id=task_id,
        instance_id="inst-1",
        command_id=CommandId("cmd-1"),
        task_params=task_params,
    )
    engine.submit(task)

    # Step 1: should generate token 1
    step_1 = list(engine.step())
    assert len(step_1) == 1
    assert step_1[0][0] == task_id

    # Step 2: should generate token 2 and finished response
    step_2 = list(engine.step())
    assert len(step_2) == 2
    assert isinstance(step_2[1][1], FinishedResponse)

    # Test hybrid fallback execution
    fallback_result = engine.hybrid_fallback_execution("gelu", np.array([-1.0, 2.0]))
    assert np.allclose(fallback_result, [0.0, 2.0])

    # Test cancellation handling
    task_cancel_params = TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content=InputMessageContent("Cancel me"))],
        max_output_tokens=2,
    )
    task_cancel = TextGeneration(
        task_id=TaskId("task-to-cancel"),
        instance_id="inst-1",
        command_id=CommandId("cmd-cancel"),
        task_params=task_cancel_params,
    )
    engine.submit(task_cancel)
    engine._cancelled_tasks.add(TaskId("task-to-cancel"))
    cancel_step = list(engine.step())
    assert len(cancel_step) == 1
    assert isinstance(cancel_step[0][1], CancelledResponse)

    engine.close()
    builder.close()


def test_npu_hybrid_fallback_execution_and_telemetry() -> None:
    """Verify hybrid delegation executes accurately for custom/dynamic activation layers."""
    profile = NpuDeviceProfile.detect_local_device()
    session = NpuExecutionSession(
        device_profile=profile,
        model_path="test-model-telemetry",
        hidden_dimension=64,
    )
    engine = NpuEngine(model_id=ModelId("test-model"), session=session)

    # Test operation compatibility
    assert session.is_operation_npu_compatible("dense_gemm") is True
    assert session.is_operation_npu_compatible("custom_triton_kernel") is False

    # Execute forward and verify telemetry token tracking
    input_mat = np.random.randn(4, 64).astype(np.float32)
    output_mat = session.execute_layer_forward(input_mat)
    assert output_mat.shape == (4, 64)
    assert session.total_tokens_executed == 4

    # Execute hybrid fallback
    raw_tensor = np.array([-3.0, -1.0, 0.0, 2.5, 5.0], dtype=np.float32)
    fallback_out = engine.hybrid_fallback_execution("dynamic_activation", raw_tensor)
    expected_out = np.array([0.0, 0.0, 0.0, 2.5, 5.0], dtype=np.float32)
    assert np.allclose(fallback_out, expected_out)

    engine.close()

