from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from exo.shared.models.model_cards import Backend, ModelCard, ModelTask
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import ExpertShardMetadata, Sharding
from exo.worker.engines.mlx.expert_parallel import (
    ExpertParallelConfiguration,
    ExpertRoutingDecision,
    ExpertTokenDispatcher,
    ShardedExpertBlock,
    SparseExpertRouter,
    combine_expert_contributions,
    compute_sparse_routing,
    expert_auto_parallel,
    softmax_stable,
)


def test_softmax_stable_numerical_correctness() -> None:
    """Verify that softmax_stable computes valid probabilities summing to 1.0."""
    logits = np.array([2.0, 1.0, 0.1, -1.0], dtype=np.float32)
    probabilities = softmax_stable(logits)
    assert np.isclose(np.sum(probabilities), 1.0, atol=1e-6)
    assert probabilities[0] > probabilities[1] > probabilities[2] > probabilities[3]


def test_sparse_expert_routing_partitioning() -> None:
    """Verify that top-k routing correctly identifies local and remote experts."""
    # 2 tokens, 8 experts
    gating_logits = np.array(
        [
            [0.1, 0.9, 0.2, 0.0, 0.8, 0.1, 0.3, 0.0],  # Top-2: expert 1 (0.9), expert 4 (0.8)
            [0.7, 0.1, 0.0, 0.6, 0.0, 0.1, 0.2, 0.0],  # Top-2: expert 0 (0.7), expert 3 (0.6)
        ],
        dtype=np.float32,
    )
    # This node only hosts experts 0 and 1
    hosted_expert_indices = [0, 1]

    decisions = compute_sparse_routing(
        gating_logits=gating_logits,
        top_k=2,
        hosted_expert_indices=hosted_expert_indices,
        normalize_weights=True,
    )

    assert len(decisions) == 2

    # Token 0: Top-2 are [1, 4] -> local: [1], remote: [4]
    assert decisions[0].selected_expert_indices == [1, 4]
    assert decisions[0].local_expert_indices == [1]
    assert decisions[0].remote_expert_indices == [4]
    assert np.isclose(sum(decisions[0].routing_weights), 1.0, atol=1e-5)

    # Token 1: Top-2 are [0, 3] -> local: [0], remote: [3]
    assert decisions[1].selected_expert_indices == [0, 3]
    assert decisions[1].local_expert_indices == [0]
    assert decisions[1].remote_expert_indices == [3]
    assert np.isclose(sum(decisions[1].routing_weights), 1.0, atol=1e-5)


def test_sparse_expert_router_forward_projection() -> None:
    """Verify that SparseExpertRouter projects hidden states to routing decisions."""
    hidden_dimension = 4
    total_experts = 4
    top_k = 2

    # Identity-like gate projection
    gate_weight = np.eye(hidden_dimension, total_experts, dtype=np.float32)
    configuration = ExpertParallelConfiguration(
        total_experts=total_experts,
        top_k_experts=top_k,
        hosted_expert_indices=[0, 2],
    )
    router = SparseExpertRouter(gate_weight=gate_weight, configuration=configuration)

    hidden_states = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],  # Highest on expert 0 (1.0) and expert 2 (0.5)
        ],
        dtype=np.float32,
    )

    decisions = router.route_tokens(hidden_states)
    assert len(decisions) == 1
    assert decisions[0].selected_expert_indices == [0, 2]
    assert decisions[0].local_expert_indices == [0, 2]
    assert len(decisions[0].remote_expert_indices) == 0


def test_combine_expert_contributions_mathematical_precision() -> None:
    """Verify that combine_expert_contributions performs accurate weighted summation."""
    decision = ExpertRoutingDecision(
        token_index=0,
        selected_expert_indices=[0, 3],
        routing_weights=[0.75, 0.25],
        local_expert_indices=[0],
        remote_expert_indices=[3],
    )
    expert_outputs = {
        0: np.array([2.0, 4.0], dtype=np.float32),
        3: np.array([6.0, 8.0], dtype=np.float32),
    }

    combined = combine_expert_contributions(
        routing_decision=decision,
        expert_outputs=expert_outputs,
        feature_dimension=2,
    )

    # 0.75 * [2, 4] + 0.25 * [6, 8] = [1.5 + 1.5, 3.0 + 2.0] = [3.0, 5.0]
    expected = np.array([3.0, 5.0], dtype=np.float32)
    np.testing.assert_allclose(combined, expected, rtol=1e-5)


def test_expert_token_dispatcher_with_local_and_remote() -> None:
    """Verify that ExpertTokenDispatcher coordinates local execution and remote dispatch."""
    configuration = ExpertParallelConfiguration(
        total_experts=4,
        top_k_experts=2,
        hosted_expert_indices=[0],  # Node only hosts expert 0
    )

    # Local expert 0: doubles the vector
    local_callables = {
        0: lambda x: x * 2.0,
    }
    sharded_block = ShardedExpertBlock(
        layer_index=0,
        configuration=configuration,
        local_expert_callables=local_callables,
    )

    # Mock gate: always picks expert 0 (score 1.0) and expert 1 (score 0.5)
    gate_weight = np.array(
        [
            [1.0, 0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    router = SparseExpertRouter(gate_weight=gate_weight, configuration=configuration)
    dispatcher = ExpertTokenDispatcher(router=router, sharded_block=sharded_block)

    # Remote handler simulating network response from node hosting expert 1
    def mock_remote_handler(
        token_index: int, token_vector: np.ndarray, remote_indices: list[int]
    ) -> dict[int, np.ndarray]:
        results: dict[int, np.ndarray] = {}
        for remote_expert in remote_indices:
            # Remote expert 1: adds 1.0 to vector
            results[remote_expert] = token_vector + 1.0
        return results

    input_tokens = np.array([[3.0, 4.0]], dtype=np.float32)
    output = dispatcher.dispatch_and_forward(
        input_tokens=input_tokens,
        remote_expert_handler=mock_remote_handler,
    )

    assert output.shape == (1, 2)
    assert not np.isnan(output).any()


def create_mock_model_card(model_id_str: str = "test-moe-model") -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id_str),
        storage_size=Memory.from_mb(1000),
        n_layers=2,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def test_expert_auto_parallel_pruning() -> None:
    """Verify that expert_auto_parallel prunes unassigned experts to save memory."""
    class MockMoEBlock:
        def __init__(self) -> None:
            self.experts = [MagicMock() for _ in range(8)]

    class MockLayer:
        def __init__(self) -> None:
            self.block_sparse_moe = MockMoEBlock()

    class MockModel:
        def __init__(self) -> None:
            self.layers = [MockLayer(), MockLayer()]

    mock_model = MockModel()
    mock_model_card = create_mock_model_card()

    # Assign only expert 1 and expert 5 to this node
    metadata = ExpertShardMetadata(
        model_card=mock_model_card,
        device_rank=0,
        world_size=4,
        start_layer=0,
        end_layer=2,
        n_layers=2,
        expert_indices=[1, 5],
        total_experts=8,
        top_k_experts=2,
    )

    # Execute generator
    generator = expert_auto_parallel(
        model=mock_model,
        group=None,
        shard_metadata=metadata,
    )
    responses = list(generator)

    assert len(responses) == 2
    assert responses[0].layers_loaded == 1
    assert responses[1].layers_loaded == 2

    # Assert that experts 1 and 5 are preserved, while 0, 2, 3, 4, 6, 7 are None
    layer_0_experts = mock_model.layers[0].block_sparse_moe.experts
    assert layer_0_experts[1] is not None
    assert layer_0_experts[5] is not None
    assert layer_0_experts[0] is None
    assert layer_0_experts[2] is None
    assert layer_0_experts[3] is None
    assert layer_0_experts[4] is None
    assert layer_0_experts[6] is None
    assert layer_0_experts[7] is None


def test_expert_shard_metadata_properties() -> None:
    """Verify ExpertShardMetadata attributes and assignment checker."""
    mock_model_card = create_mock_model_card("test-moe")
    metadata = ExpertShardMetadata(
        model_card=mock_model_card,
        device_rank=1,
        world_size=2,
        start_layer=0,
        end_layer=4,
        n_layers=16,
        expert_indices=[2, 3],
        total_experts=8,
        top_k_experts=2,
    )

    assert metadata.is_expert_assigned(2) is True
    assert metadata.is_expert_assigned(3) is True
    assert metadata.is_expert_assigned(0) is False
    assert metadata.is_expert_assigned(7) is False
    assert Sharding.Expert.value == "Expert"
