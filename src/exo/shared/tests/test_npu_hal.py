from exo.shared.types.npu_hal import (
    ComputationNode,
    FusedSubgraphDescriptor,
    NpuSessionTelemetry,
)
from exo.worker.engines.npu.subgraph_partitioner import DynamicSubgraphPartitioner
from exo.worker.engines.npu.unified_npu_hal import UnifiedNpuHalEngine


def test_npu_models_and_telemetry():
    """Verify descriptors and telemetry metrics calculations."""
    node = ComputationNode(
        node_id="norm_0",
        op_type="rmsnorm",
        inputs=("x",),
        outputs=("norm_out",),
        is_npu_supported=True,
    )

    desc = FusedSubgraphDescriptor(
        subgraph_id="sub_1",
        target_backend="directml",
        nodes=(node,),
        input_names=("x",),
        output_names=("norm_out",),
        is_fused_npu=True,
    )
    assert desc.num_ops == 1
    assert desc.is_fused_npu is True

    telemetry = NpuSessionTelemetry(
        backend_used="directml",
        total_subgraphs_executed=4,
        fused_npu_subgraphs_count=3,
        cpu_fallback_subgraphs_count=1,
        total_npu_execution_time_ms=12.5,
        total_cpu_fallback_time_ms=3.5,
    )
    assert telemetry.npu_offload_ratio == 0.75
    t_dict = telemetry.to_dict()
    assert t_dict["npu_offload_ratio_pct"] == 75.0
    assert t_dict["backend_used"] == "directml"


def test_dynamic_subgraph_partitioning():
    """Verify dynamic segmenting into fused NPU clusters and CPU fallback islands."""
    partitioner = DynamicSubgraphPartitioner()

    # Transformer Layer:
    # Cluster 1 (NPU): RMSNorm -> QKV Linear
    # Cluster 2 (CPU): RoPE -> Softmax
    # Cluster 3 (NPU): Out Linear -> Residual Add
    nodes = [
        ComputationNode(
            node_id="n1",
            op_type="rmsnorm",
            inputs=("hidden_in",),
            outputs=("norm_1",),
        ),
        ComputationNode(
            node_id="n2",
            op_type="linear",
            inputs=("norm_1", "w_qkv"),
            outputs=("qkv",),
        ),
        ComputationNode(
            node_id="n3",
            op_type="rope",
            inputs=("qkv",),
            outputs=("qkv_rot",),
            is_npu_supported=False,
        ),
        ComputationNode(
            node_id="n4",
            op_type="softmax",
            inputs=("qkv_rot",),
            outputs=("attn_weights",),
            is_npu_supported=False,
        ),
        ComputationNode(
            node_id="n5",
            op_type="linear",
            inputs=("attn_weights", "w_out"),
            outputs=("attn_out",),
        ),
        ComputationNode(
            node_id="n6",
            op_type="add",
            inputs=("hidden_in", "attn_out"),
            outputs=("layer_out",),
        ),
    ]

    subgraphs = partitioner.partition_graph(nodes, preferred_backend="directml")
    assert len(subgraphs) == 3

    # Subgraph 0: Fused NPU (n1, n2)
    assert subgraphs[0].is_fused_npu is True
    assert subgraphs[0].target_backend == "directml"
    assert len(subgraphs[0].nodes) == 2
    assert "hidden_in" in subgraphs[0].input_names
    assert "qkv" in subgraphs[0].output_names

    # Subgraph 1: Host CPU Fallback (n3, n4)
    assert subgraphs[1].is_fused_npu is False
    assert subgraphs[1].target_backend == "cpu_fallback"
    assert len(subgraphs[1].nodes) == 2
    assert "qkv" in subgraphs[1].input_names
    assert "attn_weights" in subgraphs[1].output_names

    # Subgraph 2: Fused NPU (n5, n6)
    assert subgraphs[2].is_fused_npu is True
    assert subgraphs[2].target_backend == "directml"
    assert len(subgraphs[2].nodes) == 2
    assert "layer_out" in subgraphs[2].output_names


def test_subgraph_execution_mathematical_precision():
    """Verify numeric accuracy of fused RMSNorm + Linear + SiLU + Add pipeline."""
    engine = UnifiedNpuHalEngine(override_backend="directml")

    nodes = [
        ComputationNode(
            node_id="norm",
            op_type="rmsnorm",
            inputs=("x",),
            outputs=("norm_out",),
            params={"eps": 1e-5, "gamma": [1.0, 1.0]},
        ),
        ComputationNode(
            node_id="lin",
            op_type="linear",
            inputs=("norm_out", "weight"),
            outputs=("lin_out",),
        ),
        ComputationNode(
            node_id="act",
            op_type="silu",
            inputs=("lin_out",),
            outputs=("act_out",),
        ),
        ComputationNode(
            node_id="res",
            op_type="add",
            inputs=("x", "act_out"),
            outputs=("final_out",),
        ),
    ]

    subgraph = FusedSubgraphDescriptor(
        subgraph_id="fused_dense",
        target_backend="directml",
        nodes=tuple(nodes),
        input_names=("x", "weight"),
        output_names=("final_out",),
        is_fused_npu=True,
    )

    scope = {
        "x": [3.0, 4.0],  # RMS = sqrt((9+16)/2) = sqrt(12.5) ~ 3.5355
        "weight": [
            [1.0, 0.0],
            [0.0, 1.0],
        ],  # identity matrix
    }

    res_scope = engine.execute_subgraph(subgraph, scope)
    assert "final_out" in res_scope
    assert len(res_scope["final_out"]) == 2

    # RMSNorm(3.0) ~ 3.0 / 3.5355 ~ 0.8485
    # Linear keeps it ~ 0.8485
    # SiLU(0.8485) = 0.8485 * sigmoid(0.8485) ~ 0.594
    # Final = 3.0 + 0.594 ~ 3.594
    assert 3.5 < res_scope["final_out"][0] < 3.7


def test_hybrid_layer_execution_and_telemetry():
    """Verify execution across hybrid NPU and CPU subgraphs and check session telemetry."""
    engine = UnifiedNpuHalEngine(override_backend="qnn")

    nodes = [
        ComputationNode(
            node_id="n1",
            op_type="rmsnorm",
            inputs=("x",),
            outputs=("norm_1",),
        ),
        ComputationNode(
            node_id="n2",
            op_type="rope",
            inputs=("norm_1",),
            outputs=("rot_1",),
            is_npu_supported=False,
            params={"position": 1},
        ),
        ComputationNode(
            node_id="n3",
            op_type="linear",
            inputs=("rot_1", "w"),
            outputs=("out",),
        ),
    ]

    initial_scope = {
        "x": [1.0, 2.0],
        "w": [[1.0, 0.5], [0.5, 1.0]],
    }

    final_scope, telemetry = engine.execute_layer_hybrid(nodes, initial_scope)
    assert "out" in final_scope
    assert telemetry.backend_used == "qnn"
    assert telemetry.fused_npu_subgraphs_count == 2
    assert telemetry.cpu_fallback_subgraphs_count == 1
    assert telemetry.total_subgraphs_executed == 3
    assert telemetry.npu_offload_ratio > 0.6


def test_backend_detection_and_empty_telemetry():
    """Verify fallback detection and empty telemetry edge cases."""
    engine = UnifiedNpuHalEngine(override_backend="cpu_fallback")
    assert engine.backend == "cpu_fallback"

    empty = NpuSessionTelemetry(backend_used="cpu_fallback")
    assert empty.npu_offload_ratio == 0.0
