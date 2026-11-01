"""DeepSeek V4.1's family spec, as the master's placement and the runner's bootstrap read it."""

from exo.worker.engines.mlx.models.spec_registry import (
    fast_synch_allowed,
    placement_rules_for_model_type,
)


def test_v41_runs_without_fast_synch_and_other_models_keep_it() -> None:
    assert not fast_synch_allowed("deepseek_v41")
    assert fast_synch_allowed("deepseek_v4")
    assert fast_synch_allowed(None)


def test_v41_tensor_parallelism_keeps_whole_output_groups() -> None:
    rules = placement_rules_for_model_type("deepseek_v41")
    assert rules.tensor_parallel_divides == (8,)
    assert not rules.tensor_parallel_splits_kv_heads
