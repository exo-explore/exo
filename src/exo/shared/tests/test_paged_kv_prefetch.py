import time

from exo.shared.types.paged_kv import (
    BlockTable,
    PagedKVBlock,
    PrefetchRequest,
    PrefetchTelemetry,
)
from exo.worker.engines.paged_attention.paged_cache_engine import PagedCacheEngine
from exo.worker.engines.paged_attention.prefetch_manager import (
    DoubleBufferPipeline,
    SpeculativePrefetchPredictor,
)


def test_paged_kv_block_and_block_table():
    """Verify block creation, capacity metrics, and block table mapping."""
    block = PagedKVBlock(
        block_id="blk_001",
        sequence_id="seq_test_1",
        logical_block_index=0,
        block_size=16,
        num_heads=8,
        head_dim=64,
        keys_data=((0.1, 0.2), (0.3, 0.4)),
        values_data=((0.5, 0.6), (0.7, 0.8)),
    )

    assert block.current_tokens_count == 2
    assert block.capacity_tokens == 16
    assert not block.is_full
    assert block.size_in_bytes == 2 * 8 * 64 * 2 * 4

    table = BlockTable(
        sequence_id="seq_test_1",
        block_size=16,
        logical_to_physical={0: "blk_001"},
    )
    assert table.get_block_id(0) == "blk_001"
    assert table.get_block_id(1) is None
    assert table.total_mapped_blocks() == 1
    assert table.total_capacity_tokens() == 16


def test_speculative_prefetch_predictor():
    """Verify predictive lookahead and attention sink block identification."""
    predictor = SpeculativePrefetchPredictor(
        block_size=16,
        lookahead_blocks=2,
        sink_blocks=1,
    )

    # When at token 35 (which is logical block 2)
    candidates = predictor.predict_candidate_blocks(
        current_token_index=35,
        total_tokens_in_sequence=80,
    )

    # Candidate should contain:
    # - sink block 0
    # - lookahead blocks: 2+1 = 3, 2+2 = 4
    # And should NOT contain current block 2
    assert 0 in candidates
    assert 3 in candidates
    assert 4 in candidates
    assert 2 not in candidates

    # Test with custom heavy hitters
    candidates_hh = predictor.predict_candidate_blocks(
        current_token_index=35,
        custom_heavy_hitters=[7, 9],
    )
    assert 7 in candidates_hh
    assert 9 in candidates_hh


def test_double_buffer_pipeline():
    """Verify concurrent double-buffering prefetch and compute overlap."""
    loaded_log: list[str] = []

    def mock_storage_fetch(seq_id: str, block_idx: int) -> PagedKVBlock:
        time.sleep(0.01)  # simulate 10ms I/O
        loaded_log.append(f"{seq_id}:{block_idx}")
        return PagedKVBlock(
            block_id=f"fetched_{seq_id}_{block_idx}",
            sequence_id=seq_id,
            logical_block_index=block_idx,
            block_size=16,
        )

    pipeline = DoubleBufferPipeline(storage_fetch_callback=mock_storage_fetch)

    req = PrefetchRequest(
        request_id="req_1",
        sequence_id="seq_double_buf",
        target_logical_block_indices=(1, 2),
    )

    def mock_compute(duration_sec: float) -> str:
        time.sleep(duration_sec)
        return "compute_done"

    res, comp_ms, masked_ms = pipeline.execute_with_compute_overlap(
        compute_fn=mock_compute,
        compute_args=(0.025,),
        request=req,
    )

    assert res == "compute_done"
    assert comp_ms > 0
    assert masked_ms > 0

    # Test promote to active
    block_1 = pipeline.promote_prefetch_to_active("seq_double_buf", 1)
    assert block_1 is not None
    assert block_1.logical_block_index == 1
    assert "seq_double_buf:1" in loaded_log

    pipeline.close()


def test_paged_cache_engine_full_lifecycle():
    """Verify token writing, block allocation, eviction, speculative prefetching, and telemetry."""
    engine = PagedCacheEngine(
        block_size=4,
        lookahead_blocks=2,
        sink_blocks=1,
        simulate_tier2_latency_sec=0.005,
    )

    seq_id = "test_seq_eval"

    # Write 8 tokens (occupying 2 blocks: block 0 and block 1)
    for i in range(8):
        engine.append_kv_token(
            sequence_id=seq_id,
            token_index=i,
            key_vector=(0.1 * i, 0.2 * i),
            value_vector=(0.3 * i, 0.4 * i),
            num_heads=2,
            head_dim=2,
        )

    table = engine.get_or_create_block_table(seq_id)
    assert table.total_mapped_blocks() == 2

    # Verify initial read hit
    blk_0, hit = engine.read_block(seq_id, 0)
    assert hit is True
    assert blk_0 is not None
    assert blk_0.current_tokens_count == 4

    # Evict block 1 to secondary tier (SSD)
    evicted = engine.evict_block_to_secondary_tier(seq_id, 1, target_location="nvme_ssd")
    assert evicted is True
    assert table.get_block_id(1) not in engine.fast_pool

    # Forward step at token 0 (which predicts block 0 as sink, block 1 as lookahead)
    blk_res, telemetry = engine.step_forward_with_prefetch(
        sequence_id=seq_id,
        current_token_index=0,
    )

    assert blk_res is not None
    assert blk_res.logical_block_index == 0
    assert telemetry.total_prefetches > 0
    assert telemetry.io_time_masked_ms > 0

    # Now read block 1 - it should have been prefetched back into fast tier!
    blk_1_prefetched, hit_1 = engine.read_block(seq_id, 1)
    assert blk_1_prefetched is not None
    assert hit_1 is True

    telemetry_dict = engine.get_telemetry().to_dict()
    assert "hit_rate_pct" in telemetry_dict
    assert "masking_efficiency_pct" in telemetry_dict

    engine.close()


def test_telemetry_properties():
    """Verify edge cases in PrefetchTelemetry calculations."""
    empty_telemetry = PrefetchTelemetry()
    assert empty_telemetry.hit_rate == 0.0
    assert empty_telemetry.masking_efficiency == 0.0

    telemetry = PrefetchTelemetry(
        total_prefetches=10,
        cache_hits=8,
        cache_misses=2,
        io_time_masked_ms=25.0,
        total_io_latency_ms=25.0,
    )
    assert telemetry.hit_rate == 0.8
    assert telemetry.masking_efficiency == 1.0
    t_dict = telemetry.to_dict()
    assert t_dict["hit_rate_pct"] == 80.0
    assert t_dict["masking_efficiency_pct"] == 100.0
