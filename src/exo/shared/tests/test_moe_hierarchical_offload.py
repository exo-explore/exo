import time

from exo.shared.types.moe_offload import (
    MoEOffloadTelemetry,
    OffloadedExpertWeight,
)
from exo.worker.engines.moe.hierarchical_expert_manager import (
    EarlyLayerPrefetchPredictor,
    HierarchicalExpertEngine,
)


def test_offloaded_expert_weight_properties():
    """Verify expert data container, memory sizing, and key formatting."""
    weights = (0.1, -0.2, 0.3, -0.4, 0.5)
    expert = OffloadedExpertWeight(
        expert_id=3,
        layer_id=2,
        num_params=len(weights),
        size_in_bytes=len(weights) * 4,
        current_tier="host_ram",
        is_pinned_hot=True,
        weights_data=weights,
    )

    assert expert.expert_key == "l2_e3"
    assert expert.num_params == 5
    assert expert.size_in_bytes == 20
    assert expert.is_pinned_hot is True
    assert expert.current_tier == "host_ram"


def test_early_layer_prefetch_predictor():
    """Verify predictive targeting of layer L+4 and Top-K extraction."""
    predictor = EarlyLayerPrefetchPredictor(lookahead_layers=4)

    # List input: 8 experts with probabilities
    router_probs = [0.05, 0.40, 0.10, 0.05, 0.25, 0.05, 0.05, 0.05]
    predicted = predictor.predict_future_experts(
        current_layer_id=0,
        router_probabilities=router_probs,
        top_k=2,
    )

    # Top-2 experts are index 1 (0.40) and index 4 (0.25) at layer 0+4=4
    assert predicted == [(4, 1), (4, 4)]

    # Dict input
    dict_probs = {0: 0.1, 1: 0.7, 2: 0.2}
    predicted_dict = predictor.predict_future_experts(
        current_layer_id=2,
        router_probabilities=dict_probs,
        top_k=1,
    )
    assert predicted_dict == [(6, 1)]


def test_hierarchical_expert_hot_and_cold_requests():
    """Verify fast pool hits vs cold storage fallback."""
    engine = HierarchicalExpertEngine(fast_capacity_slots=3, simulate_ssd_latency_sec=0.002)

    # Register 1 hot pinned expert and 1 cold SSD expert
    engine.register_expert(layer_id=0, expert_id=1, weights_data=[1.0, 2.0], tier="host_ram", is_pinned_hot=True)
    engine.register_expert(layer_id=1, expert_id=2, weights_data=[3.0, 4.0], tier="nvme_ssd", is_pinned_hot=False)

    # 1. Hot request
    exp_hot, status_hot = engine.request_expert(0, 1)
    assert status_hot == "hot_hit"
    assert exp_hot is not None
    assert exp_hot.is_pinned_hot is True

    # 2. Cold request
    exp_cold, status_cold = engine.request_expert(1, 2)
    assert status_cold == "cold_miss"
    assert exp_cold is not None
    assert exp_cold.current_tier == "host_ram"

    engine.close()


def test_lru_unpinned_eviction_when_capacity_exceeded():
    """Verify that unpinned experts are evicted to SSD when capacity is reached, while pinned experts stay."""
    engine = HierarchicalExpertEngine(fast_capacity_slots=2, simulate_ssd_latency_sec=0.001)

    # Slot 1: Pinned Hot Expert
    engine.register_expert(layer_id=0, expert_id=0, weights_data=[1.0], tier="host_ram", is_pinned_hot=True)

    # Cold experts to load
    engine.register_expert(layer_id=1, expert_id=1, weights_data=[2.0], tier="nvme_ssd")
    engine.register_expert(layer_id=2, expert_id=2, weights_data=[3.0], tier="nvme_ssd")

    # Load first cold expert (now fast pool has: l0_e0, l1_e1)
    engine.request_expert(1, 1)
    assert "l1_e1" in engine.fast_pool
    assert "l0_e0" in engine.fast_pool

    # Load second cold expert (fast pool exceeds 2: must evict unpinned l1_e1)
    time.sleep(0.005)
    engine.request_expert(2, 2)
    assert "l2_e2" in engine.fast_pool
    assert "l0_e0" in engine.fast_pool  # Pinned never evicted!
    assert "l1_e1" not in engine.fast_pool
    assert "l1_e1" in engine.cold_storage
    assert engine.cold_storage["l1_e1"].current_tier == "nvme_ssd"

    engine.close()


def test_early_prefetch_and_overlap_lifecycle():
    """Verify predictive early prefetching masking SSD read latency."""
    engine = HierarchicalExpertEngine(fast_capacity_slots=4, simulate_ssd_latency_sec=0.005)

    # Register layer 4 cold experts
    engine.register_expert(layer_id=4, expert_id=1, weights_data=[1.1, 2.2], tier="nvme_ssd")
    engine.register_expert(layer_id=4, expert_id=2, weights_data=[3.3, 4.4], tier="nvme_ssd")

    # Register layer 0 active expert
    engine.register_expert(layer_id=0, expert_id=0, weights_data=[0.0, 0.0], tier="host_ram")

    # Execute step at layer 0 with router predicting layer 4 expert 1
    router_probs = [0.0, 0.9, 0.1]  # expert 1 has 0.9 probability
    def mock_compute():
        time.sleep(0.015)  # 15ms compute work

    exp_active, telemetry = engine.step_forward_with_early_prefetch(
        current_layer_id=0,
        current_expert_id=0,
        router_probabilities=router_probs,
        compute_fn=mock_compute,
    )

    assert exp_active is not None
    assert exp_active.expert_key == "l0_e0"
    assert telemetry.io_time_masked_ms > 0

    # Advance to layer 4 and request expert 1: it was already prefetched!
    exp_4_1, status = engine.request_expert(layer_id=4, expert_id=1)
    assert status == "prefetch_hit"
    assert exp_4_1 is not None

    final_telemetry = engine.get_telemetry()
    assert final_telemetry.speculative_prefetch_hits == 1
    assert final_telemetry.hit_rate > 0.0
    t_dict = final_telemetry.to_dict()
    assert t_dict["hit_rate_pct"] > 0.0

    engine.close()


def test_telemetry_properties():
    """Verify edge cases in telemetry calculations."""
    empty = MoEOffloadTelemetry()
    assert empty.hit_rate == 0.0
    assert empty.prefetch_masking_ratio == 0.0
