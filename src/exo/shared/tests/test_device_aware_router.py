import numpy as np
import pytest

from exo.routing.device_aware_router import DeviceAwareMoERouter
from exo.shared.types.common import NodeId
from exo.shared.types.moe_routing import NodeComputeProfile


def test_homogeneous_cluster_routing() -> None:
    node1 = NodeId("node-pc-1")
    node2 = NodeId("node-pc-2")
    profiles = {
        node1: NodeComputeProfile(node_id=node1, compute_weight=1.0, queue_depth=0),
        node2: NodeComputeProfile(node_id=node2, compute_weight=1.0, queue_depth=0),
    }
    placements = {0: node1, 1: node1, 2: node2, 3: node2}

    np.random.seed(42)
    num_tokens, num_experts = 16, 4
    logits = np.random.randn(num_tokens, num_experts).astype(np.float32)

    router = DeviceAwareMoERouter(alpha=0.5, beta=0.5, gamma=1.0)
    decision, telemetry = router.route_tokens(logits, profiles, placements, top_k=2)

    assert telemetry.total_tokens == num_tokens
    assert telemetry.overflow_count == 0
    assert len(decision.selected_experts) == num_tokens

    # Verify that each token received 2 experts and weights sum to 1.0
    for exp_list, weight_list in zip(decision.selected_experts, decision.routing_weights, strict=True):
        assert len(exp_list) == 2
        assert sum(weight_list) == pytest.approx(1.0, abs=1e-5)


def test_heterogeneous_compute_weight_bias() -> None:
    gpu_node = NodeId("gpu-desktop")
    phone_node = NodeId("mobile-phone")
    profiles = {
        gpu_node: NodeComputeProfile(node_id=gpu_node, compute_weight=3.0, queue_depth=0),
        phone_node: NodeComputeProfile(node_id=phone_node, compute_weight=0.2, queue_depth=0),
    }
    placements = {0: gpu_node, 1: gpu_node, 2: phone_node, 3: phone_node}

    # Flat logits so routing preference is driven purely by compute profile
    num_tokens = 32
    logits = np.zeros((num_tokens, 4), dtype=np.float32)

    router = DeviceAwareMoERouter(alpha=0.5, beta=1.0, gamma=1.0)
    decision, telemetry = router.route_tokens(logits, profiles, placements, top_k=2)

    gpu_tokens = decision.node_token_counts[str(gpu_node)]
    phone_tokens = decision.node_token_counts[str(phone_node)]

    # GPU should take significantly higher proportion of workload
    assert gpu_tokens > phone_tokens
    assert telemetry.load_balance_cv >= 0.0


def test_queue_depth_penalty() -> None:
    node1 = NodeId("congested-node")
    node2 = NodeId("idle-node")
    profiles = {
        node1: NodeComputeProfile(node_id=node1, compute_weight=1.0, queue_depth=30),  # Heavy backlog
        node2: NodeComputeProfile(node_id=node2, compute_weight=1.0, queue_depth=0),   # Zero backlog
    }
    placements = {0: node1, 1: node1, 2: node2, 3: node2}

    num_tokens = 20
    logits = np.zeros((num_tokens, 4), dtype=np.float32)

    router = DeviceAwareMoERouter(alpha=1.0, beta=0.5, gamma=1.0)
    decision, _ = router.route_tokens(logits, profiles, placements, top_k=2)

    node1_tokens = decision.node_token_counts[str(node1)]
    node2_tokens = decision.node_token_counts[str(node2)]

    # Congested node should receive fewer tokens than idle node
    assert node2_tokens > node1_tokens


def test_thermal_throttling_avoidance() -> None:
    cool_node = NodeId("cool-node")
    throttled_node = NodeId("hot-node")
    profiles = {
        cool_node: NodeComputeProfile(node_id=cool_node, compute_weight=1.0, is_throttled=False),
        throttled_node: NodeComputeProfile(node_id=throttled_node, compute_weight=1.0, is_throttled=True),
    }
    placements = {0: cool_node, 1: cool_node, 2: throttled_node, 3: throttled_node}

    num_tokens = 16
    logits = np.zeros((num_tokens, 4), dtype=np.float32)

    router = DeviceAwareMoERouter(alpha=0.5, beta=0.5, gamma=2.0)
    decision, _ = router.route_tokens(logits, profiles, placements, top_k=2)

    assert decision.node_token_counts[str(cool_node)] > decision.node_token_counts[str(throttled_node)]


def test_capacity_overflow_and_weight_normalization() -> None:
    small_node = NodeId("small-node")
    fallback_node = NodeId("fallback-node")
    profiles = {
        # Strict capacity limit of 4 tokens on small node
        small_node: NodeComputeProfile(node_id=small_node, compute_weight=1.0, max_capacity=4),
        fallback_node: NodeComputeProfile(node_id=fallback_node, compute_weight=1.0, max_capacity=100),
    }
    placements = {0: small_node, 1: fallback_node}

    # All 10 tokens strongly prefer Expert 0 on small node
    logits = np.array([[10.0, -10.0]] * 10, dtype=np.float32)

    router = DeviceAwareMoERouter(capacity_factor=1.5)
    decision, telemetry = router.route_tokens(logits, profiles, placements, top_k=1)
    assert telemetry.total_tokens == 10

    # Small node should not exceed its capacity
    assert decision.node_token_counts[str(small_node)] <= 4
    # Overflow tokens should have fallen back to Expert 1 on fallback node
    assert decision.node_token_counts[str(fallback_node)] >= 6

    for w_list in decision.routing_weights:
        if w_list:
            assert sum(w_list) == pytest.approx(1.0, abs=1e-5)
