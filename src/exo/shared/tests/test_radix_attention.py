from exo.shared.types.radix_cache import (
    RadixTelemetry,
    compute_token_hash,
)
from exo.worker.engines.paged_attention.radix_tree import RadixTreeEngine


def test_radix_hash_computation():
    """Verify deterministic SHA256 content addressable hash."""
    h1 = compute_token_hash((101, 102, 103))
    h2 = compute_token_hash((101, 102, 103))
    h3 = compute_token_hash((101, 102, 104))

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64


def test_radix_tree_simple_insertion_and_exact_match():
    """Verify insertion and prefix matching for single branch."""
    engine = RadixTreeEngine()

    tokens = [1, 2, 3, 4, 5]
    blocks = ["blk_1", "blk_2"]
    node_id = engine.insert_sequence(tokens, blocks)

    assert node_id in engine.nodes

    # Exact match
    res = engine.match_prefix(tokens)
    assert res.is_full_match is True
    assert res.matched_tokens_count == 5
    assert res.matched_block_ids == ("blk_1", "blk_2")
    assert res.remaining_tokens == ()

    # Partial prefix match
    res_partial = engine.match_prefix([1, 2, 3, 99, 100])
    assert res_partial.is_full_match is False
    assert res_partial.matched_tokens_count == 3
    assert res_partial.remaining_tokens == (99, 100)


def test_radix_tree_dynamic_node_splitting():
    """Verify automatic node splitting when inserting branching shared prefix."""
    engine = RadixTreeEngine()

    # Shared system prompt: [100, 200, 300]
    # Sequence A: [100, 200, 300, 401, 402]
    # Sequence B: [100, 200, 300, 501, 502]
    engine.insert_sequence([100, 200, 300, 401, 402], ["blk_A1", "blk_A2"])
    engine.insert_sequence([100, 200, 300, 501, 502], ["blk_B1", "blk_B2"])

    # Match Sequence A
    res_a = engine.match_prefix([100, 200, 300, 401, 402])
    assert res_a.is_full_match is True
    assert res_a.matched_tokens_count == 5

    # Match Sequence B
    res_b = engine.match_prefix([100, 200, 300, 501, 502])
    assert res_b.is_full_match is True
    assert res_b.matched_tokens_count == 5

    # Match shared prefix only
    res_shared = engine.match_prefix([100, 200, 300, 999])
    assert res_shared.is_full_match is False
    assert res_shared.matched_tokens_count == 3
    assert res_shared.remaining_tokens == (999,)


def test_radix_tree_lru_eviction():
    """Verify LRU leaf node pruning for unpinned, unreferenced branches."""
    engine = RadixTreeEngine()

    node_a = engine.insert_sequence([1, 2, 3], ["blk_evict_1"])
    # Set reference count to 0 to simulate completion
    victim_node = engine.nodes[node_a]
    engine.nodes[node_a] = victim_node.model_copy(update={"reference_count": 0, "is_pinned": False})

    evicted_blocks = engine.evict_lru_branches(max_evictions=1)
    assert "blk_evict_1" in evicted_blocks
    assert node_a not in engine.nodes


def test_radix_telemetry():
    """Verify telemetry tracking for query hits and avoided token computation."""
    engine = RadixTreeEngine()
    engine.record_query(matched_tokens=80, computed_tokens=20)
    engine.record_query(matched_tokens=0, computed_tokens=50)

    telemetry = engine.get_telemetry()
    assert telemetry.total_queries == 2
    assert telemetry.total_prefix_hits == 1
    assert telemetry.hit_rate == 0.5
    assert telemetry.total_tokens_avoided == 80
    assert telemetry.total_tokens_computed == 70
    assert round(telemetry.token_savings_ratio, 3) == round(80 / 150, 3)

    t_dict = telemetry.to_dict()
    assert t_dict["hit_rate_pct"] == 50.0
    assert t_dict["total_tokens_avoided"] == 80


def test_empty_radix_telemetry():
    """Verify safe zero handling in telemetry."""
    empty = RadixTelemetry()
    assert empty.hit_rate == 0.0
    assert empty.token_savings_ratio == 0.0
