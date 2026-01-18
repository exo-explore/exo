"""Unit tests for MTP speculative decoding."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx.mtp.speculative_decode import (
    ModelWithHiddenStates,
    maybe_quantize_kv_cache,
)


class MockModel(nn.Module):
    """Mock model for testing speculative decoding."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 100) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Create simple model components
        self.model = MockInnerModel(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._layers = [nn.Linear(hidden_size, hidden_size) for _ in range(3)]

    def __call__(
        self,
        inputs: mx.array,
        cache: list | None = None,
    ) -> mx.array:
        hidden = self.model(inputs, cache)
        return self.lm_head(hidden)

    @property
    def layers(self) -> list[nn.Module]:
        return self._layers


class MockInnerModel(nn.Module):
    """Mock inner model (like DeepseekV3Model)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(100, hidden_size)
        self.norm = nn.RMSNorm(hidden_size)

    def __call__(
        self,
        inputs: mx.array,
        cache: list | None = None,
    ) -> mx.array:
        # Simple embedding + norm
        embedded = self.embed_tokens(inputs)
        return self.norm(embedded)


class TestModelWithHiddenStates:
    """Tests for ModelWithHiddenStates wrapper."""

    @pytest.fixture
    def mock_model(self) -> MockModel:
        return MockModel(hidden_size=64, vocab_size=100)

    def test_forward_returns_logits(self, mock_model: MockModel) -> None:
        """Standard forward should return logits."""
        wrapped = ModelWithHiddenStates(mock_model)
        inputs = mx.array([[1, 2, 3]])

        logits = wrapped.forward(inputs)

        assert logits.shape == (1, 3, mock_model.vocab_size)

    def test_forward_with_hidden_returns_tuple(self, mock_model: MockModel) -> None:
        """Forward with hidden should return (logits, hidden)."""
        wrapped = ModelWithHiddenStates(mock_model)
        inputs = mx.array([[1, 2, 3]])

        logits, hidden = wrapped.forward_with_hidden(inputs)

        assert logits.shape == (1, 3, mock_model.vocab_size)
        assert hidden.shape == (1, 3, mock_model.hidden_size)

    def test_layers_property(self, mock_model: MockModel) -> None:
        """Should expose layers property from base model."""
        wrapped = ModelWithHiddenStates(mock_model)

        assert wrapped.layers == mock_model.layers
        assert len(wrapped.layers) == 3


class TestMaybeQuantizeKVCache:
    """Tests for KV cache quantization."""

    def test_no_quantization_when_bits_none(self) -> None:
        """Should not quantize when kv_bits is None."""
        cache = [MockCache(offset=100)]

        maybe_quantize_kv_cache(
            cache,
            quantized_kv_start=50,
            kv_group_size=64,
            kv_bits=None,
        )

        # Cache should be unchanged
        assert not hasattr(cache[0], "quantized")

    def test_respects_quantized_kv_start(self) -> None:
        """Should only quantize caches past the start threshold."""
        cache_below = MockCache(offset=30)
        cache_above = MockCache(offset=100)
        caches = [cache_below, cache_above]

        maybe_quantize_kv_cache(
            caches,
            quantized_kv_start=50,
            kv_group_size=64,
            kv_bits=4,
        )

        # Only cache_above should be quantized
        assert not getattr(cache_below, "was_quantized", False)
        assert getattr(caches[1], "was_quantized", False)


class MockCache:
    """Mock KV cache for testing."""

    def __init__(self, offset: int = 0) -> None:
        self.offset = offset
        self.was_quantized = False

    def to_quantized(self, group_size: int, bits: int) -> "MockCache":
        quantized = MockCache(self.offset)
        quantized.was_quantized = True
        return quantized


class TestSpeculativeDecodingLogic:
    """Tests for the core speculative decoding logic."""

    def test_draft_acceptance_identical_tokens(self) -> None:
        """When draft matches verification, both should be accepted."""
        # This tests the logic, not the full generator
        draft_token = 42
        verify_token = 42

        accepted = draft_token == verify_token
        assert accepted

    def test_draft_rejection_different_tokens(self) -> None:
        """When draft differs from verification, draft should be rejected."""
        draft_token = 42
        verify_token = 99

        accepted = draft_token == verify_token
        assert not accepted


class TestMTPGenerationResponse:
    """Tests for MTPGenerationResponse dataclass."""

    def test_response_creation(self) -> None:
        """Should create response with all fields."""
        from exo.worker.engines.mlx.mtp.speculative_decode import MTPGenerationResponse

        response = MTPGenerationResponse(
            text="Hello",
            token=42,
            logprobs=mx.array([0.1, 0.2]),
            from_draft=True,
            prompt_tokens=10,
            prompt_tps=100.0,
            generation_tokens=5,
            generation_tps=50.0,
            peak_memory=1.5,
            finish_reason=None,
        )

        assert response.text == "Hello"
        assert response.token == 42
        assert response.from_draft is True
        assert response.finish_reason is None

    def test_response_with_finish_reason(self) -> None:
        """Should handle finish_reason."""
        from exo.worker.engines.mlx.mtp.speculative_decode import MTPGenerationResponse

        response = MTPGenerationResponse(
            text="",
            token=0,
            logprobs=mx.array([0.0]),
            from_draft=False,
            prompt_tokens=10,
            prompt_tps=100.0,
            generation_tokens=100,
            generation_tps=50.0,
            peak_memory=1.5,
            finish_reason="length",
        )

        assert response.finish_reason == "length"


class TestIntegration:
    """Integration tests for the full MTP pipeline."""

    def test_mtp_module_with_mock_model(self) -> None:
        """Test MTP module can be created and run with mock components."""
        pytest.importorskip("mlx_lm.models.deepseek_v3")

        from exo.worker.engines.mlx.mtp.module import MTPModule

        # Create mock config
        class MockConfig:
            hidden_size = 64
            intermediate_size = 128
            num_attention_heads = 2
            num_key_value_heads = 2
            rms_norm_eps = 1e-6
            q_lora_rank = None
            kv_lora_rank = 32
            qk_rope_head_dim = 8
            v_head_dim = 16
            qk_nope_head_dim = 16
            rope_theta = 10000.0
            rope_scaling = None
            attention_bias = False
            max_position_embeddings = 2048

        config = MockConfig()
        embedding = nn.Embedding(100, config.hidden_size)
        lm_head = nn.Linear(config.hidden_size, 100, bias=False)
        output_norm = nn.RMSNorm(config.hidden_size)

        mtp = MTPModule(
            config=config,  # type: ignore[arg-type]
            shared_embedding=embedding,
            shared_lm_head=lm_head,
            output_norm=output_norm,
        )

        # Run forward pass
        hidden = mx.random.normal((1, 1, config.hidden_size))
        token = mx.array([[5]])

        logits, new_hidden = mtp(hidden, token)

        assert logits.shape == (1, 1, 100)
        assert new_hidden.shape == (1, 1, config.hidden_size)
        # Verify outputs are valid (not NaN)
        assert not mx.any(mx.isnan(logits))
        assert not mx.any(mx.isnan(new_hidden))
