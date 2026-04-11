"""Tests for asymmetric tensor parallelism ratio finding and sharding."""



class TestFindValidRatios:
    """Test the ratio solver that finds valid asymmetric split points."""

    def test_qwen3_5_122b_dimensions(self) -> None:
        from exo.worker.engines.mlx.asymmetric_parallel import find_valid_ratios

        ratios = find_valid_ratios(
            memory_fractions=[0.73, 0.27],
            hidden_size=3072,
            num_attention_heads=32,
            num_key_value_heads=2,
            linear_num_value_heads=64,
            linear_num_key_heads=16,
            moe_intermediate_size=1024,
            num_experts=256,
        )
        assert ratios is not None
        assert len(ratios) == 2
        assert abs(ratios[0] + ratios[1] - 1.0) < 1e-10
        # All head counts must be exact integers after split
        assert 32 * ratios[0] == int(32 * ratios[0])  # attention heads
        assert 64 * ratios[0] == int(64 * ratios[0])  # value heads
        assert 16 * ratios[0] == int(16 * ratios[0])  # key heads

    def test_llama_70b_dimensions(self) -> None:
        from exo.worker.engines.mlx.asymmetric_parallel import find_valid_ratios

        ratios = find_valid_ratios(
            memory_fractions=[0.73, 0.27],
            hidden_size=8192,
            num_attention_heads=64,
            num_key_value_heads=8,
        )
        assert ratios is not None
        assert 64 * ratios[0] == int(64 * ratios[0])

    def test_nemotron_120b_dimensions(self) -> None:
        from exo.worker.engines.mlx.asymmetric_parallel import find_valid_ratios

        ratios = find_valid_ratios(
            memory_fractions=[0.73, 0.27],
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=2,
        )
        assert ratios is not None
        assert 32 * ratios[0] == int(32 * ratios[0])

    def test_rejects_impossible_dimensions(self) -> None:
        """Prime-number head count with no valid fractional split."""
        from exo.worker.engines.mlx.asymmetric_parallel import find_valid_ratios

        ratios = find_valid_ratios(
            memory_fractions=[0.73, 0.27],
            hidden_size=3072,
            num_attention_heads=7,  # prime — can't split into 2 integer parts > 0.5
            num_key_value_heads=2,
        )
        assert ratios is None

    def test_only_two_nodes_supported(self) -> None:
        from exo.worker.engines.mlx.asymmetric_parallel import find_valid_ratios

        ratios = find_valid_ratios(
            memory_fractions=[0.5, 0.25, 0.25],
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        assert ratios is None

    def test_ratio_closer_to_target(self) -> None:
        """Ratio should be the closest valid one to the memory fraction."""
        from exo.worker.engines.mlx.asymmetric_parallel import find_valid_ratios

        # With 80% target, 0.8125 (13/16) is closer than 0.75 (12/16)
        ratios = find_valid_ratios(
            memory_fractions=[0.80, 0.20],
            hidden_size=3072,
            num_attention_heads=32,
            num_key_value_heads=2,
            linear_num_value_heads=64,
            linear_num_key_heads=16,
        )
        assert ratios is not None
        assert abs(ratios[0] - 0.80) < abs(0.75 - 0.80)

    def test_equal_memory_returns_near_symmetric(self) -> None:
        """When memory is roughly equal, ratio should be close to 0.5."""
        from exo.worker.engines.mlx.asymmetric_parallel import find_valid_ratios

        ratios = find_valid_ratios(
            memory_fractions=[0.50, 0.50],
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        # Finder searches > 0.5, so it may find a near-symmetric split
        if ratios is not None:
            assert ratios[0] < 0.6  # should be close to 0.5
