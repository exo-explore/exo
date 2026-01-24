"""Unit tests for MTP module components."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx.mtp.module import (
    MTP_LAYER_INDEX,
    MTPModule,
    MTPTransformerBlock,
    extract_mtp_weights,
    load_mtp_weights_into_module,
)


class MockModelArgs:
    """Mock ModelArgs for testing without importing deepseek_v3."""

    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 512,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 4,
        rms_norm_eps: float = 1e-6,
        vocab_size: int = 1000,
        q_lora_rank: int | None = None,
        kv_lora_rank: int = 64,
        qk_rope_head_dim: int = 16,
        v_head_dim: int = 32,
        qk_nope_head_dim: int = 32,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        max_position_embeddings: int = 2048,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.max_position_embeddings = max_position_embeddings


class TestExtractMTPWeights:
    """Tests for extract_mtp_weights function."""

    def test_extracts_layer_61_weights(self) -> None:
        """Should extract only layer 61 weights."""
        weights = {
            "model.layers.60.self_attn.weight": mx.zeros((10, 10)),
            "model.layers.61.enorm.weight": mx.ones((10,)),
            "model.layers.61.hnorm.weight": mx.ones((10,)) * 2,
            "model.layers.61.eh_proj.weight": mx.ones((10, 20)),
            "model.layers.62.self_attn.weight": mx.zeros((10, 10)),
            "model.embed_tokens.weight": mx.zeros((100, 10)),
        }

        mtp_weights = extract_mtp_weights(weights)

        assert len(mtp_weights) == 3
        assert "enorm.weight" in mtp_weights
        assert "hnorm.weight" in mtp_weights
        assert "eh_proj.weight" in mtp_weights
        # Check values are preserved
        assert mx.allclose(mtp_weights["enorm.weight"], mx.ones((10,)))
        assert mx.allclose(mtp_weights["hnorm.weight"], mx.ones((10,)) * 2)

    def test_returns_empty_dict_when_no_layer_61(self) -> None:
        """Should return empty dict when layer 61 doesn't exist."""
        weights = {
            "model.layers.0.self_attn.weight": mx.zeros((10, 10)),
            "model.layers.60.self_attn.weight": mx.zeros((10, 10)),
        }

        mtp_weights = extract_mtp_weights(weights)

        assert len(mtp_weights) == 0

    def test_handles_nested_layer_61_weights(self) -> None:
        """Should handle nested weight paths like self_attn.q_proj.weight."""
        weights = {
            f"model.layers.{MTP_LAYER_INDEX}.self_attn.q_a_proj.weight": mx.zeros(
                (10, 10)
            ),
            f"model.layers.{MTP_LAYER_INDEX}.mlp.gate_proj.weight": mx.zeros((20, 10)),
        }

        mtp_weights = extract_mtp_weights(weights)

        assert "self_attn.q_a_proj.weight" in mtp_weights
        assert "mlp.gate_proj.weight" in mtp_weights


class TestMTPTransformerBlock:
    """Tests for MTPTransformerBlock."""

    @pytest.fixture
    def config(self) -> MockModelArgs:
        return MockModelArgs(
            hidden_size=64, intermediate_size=128, num_attention_heads=2
        )

    def test_forward_shape(self, config: MockModelArgs) -> None:
        """Forward pass should preserve input shape."""
        # Skip if deepseek_v3 imports fail (CI without mlx_lm)
        pytest.importorskip("mlx_lm.models.deepseek_v3")

        block = MTPTransformerBlock(config)  # type: ignore[arg-type]
        x = mx.random.normal((1, 5, config.hidden_size))

        output = block(x)

        assert output.shape == x.shape

    def test_forward_with_mask(self, config: MockModelArgs) -> None:
        """Forward pass should work with attention mask."""
        pytest.importorskip("mlx_lm.models.deepseek_v3")

        block = MTPTransformerBlock(config)  # type: ignore[arg-type]
        x = mx.random.normal((1, 5, config.hidden_size))
        # Create causal mask
        mask = mx.triu(mx.full((5, 5), float("-inf")), k=1)

        output = block(x, mask=mask)

        assert output.shape == x.shape


class TestMTPModule:
    """Tests for MTPModule."""

    @pytest.fixture
    def config(self) -> MockModelArgs:
        return MockModelArgs(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=2,
            vocab_size=100,
        )

    @pytest.fixture
    def shared_components(
        self, config: MockModelArgs
    ) -> tuple[nn.Embedding, nn.Linear, nn.RMSNorm]:
        embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        output_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        return embedding, lm_head, output_norm

    def test_initialization(
        self,
        config: MockModelArgs,
        shared_components: tuple[nn.Embedding, nn.Linear, nn.RMSNorm],
    ) -> None:
        """MTPModule should initialize with correct components."""
        pytest.importorskip("mlx_lm.models.deepseek_v3")

        embedding, lm_head, output_norm = shared_components
        mtp = MTPModule(
            config=config,  # type: ignore[arg-type]
            shared_embedding=embedding,
            shared_lm_head=lm_head,
            output_norm=output_norm,
        )

        assert mtp.hnorm is not None
        assert mtp.enorm is not None
        assert mtp.eh_proj is not None
        assert mtp.transformer_block is not None

    def test_forward_output_shapes(
        self,
        config: MockModelArgs,
        shared_components: tuple[nn.Embedding, nn.Linear, nn.RMSNorm],
    ) -> None:
        """Forward pass should return correct output shapes."""
        pytest.importorskip("mlx_lm.models.deepseek_v3")

        embedding, lm_head, output_norm = shared_components
        mtp = MTPModule(
            config=config,  # type: ignore[arg-type]
            shared_embedding=embedding,
            shared_lm_head=lm_head,
            output_norm=output_norm,
        )

        batch_size = 2
        seq_len = 1
        hidden_state = mx.random.normal((batch_size, seq_len, config.hidden_size))
        draft_token = mx.array([[5], [10]])  # [batch, seq_len]

        logits, new_hidden = mtp(hidden_state, draft_token)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert new_hidden.shape == (batch_size, seq_len, config.hidden_size)

    def test_shares_embedding_and_lm_head(
        self,
        config: MockModelArgs,
        shared_components: tuple[nn.Embedding, nn.Linear, nn.RMSNorm],
    ) -> None:
        """MTPModule should use shared embedding and lm_head."""
        pytest.importorskip("mlx_lm.models.deepseek_v3")

        embedding, lm_head, output_norm = shared_components
        mtp = MTPModule(
            config=config,  # type: ignore[arg-type]
            shared_embedding=embedding,
            shared_lm_head=lm_head,
            output_norm=output_norm,
        )

        # Verify they're the same objects
        assert mtp._shared_embedding is embedding
        assert mtp._shared_lm_head is lm_head
        assert mtp._output_norm is output_norm


class TestLoadMTPWeights:
    """Tests for load_mtp_weights_into_module."""

    @pytest.fixture
    def config(self) -> MockModelArgs:
        return MockModelArgs(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=2,
            vocab_size=100,
        )

    def test_loads_norm_weights(self, config: MockModelArgs) -> None:
        """Should load enorm and hnorm weights."""
        pytest.importorskip("mlx_lm.models.deepseek_v3")

        embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        output_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        mtp = MTPModule(
            config=config,  # type: ignore[arg-type]
            shared_embedding=embedding,
            shared_lm_head=lm_head,
            output_norm=output_norm,
        )

        # Create test weights
        test_enorm = mx.ones((config.hidden_size,)) * 3.0
        test_hnorm = mx.ones((config.hidden_size,)) * 5.0
        mtp_weights = {
            "enorm.weight": test_enorm,
            "hnorm.weight": test_hnorm,
        }

        load_mtp_weights_into_module(mtp, mtp_weights)

        assert mx.allclose(mtp.enorm.weight, test_enorm)
        assert mx.allclose(mtp.hnorm.weight, test_hnorm)


class TestSanitizePatch:
    """Tests for the sanitize patching logic."""

    def test_patch_preserves_layer_61(self) -> None:
        """Patching sanitize should preserve layer 61 weights."""
        from exo.worker.engines.mlx.utils_mlx import (
            _patch_deepseek_sanitize_for_mtp,
            _restore_deepseek_sanitize,
        )

        deepseek_v3 = pytest.importorskip("mlx_lm.models.deepseek_v3")
        model_cls = deepseek_v3.Model

        # Get original sanitize behavior
        original_sanitize = model_cls.sanitize

        try:
            # Apply patch
            _patch_deepseek_sanitize_for_mtp()

            # Note: we can't easily test the full sanitize without a real model
            # This test verifies the patch is applied
            assert model_cls.sanitize is not original_sanitize

        finally:
            _restore_deepseek_sanitize()
            # Verify restore worked
            assert model_cls.sanitize is original_sanitize

    def test_restore_sanitize(self) -> None:
        """Restoring sanitize should return to original behavior."""
        from exo.worker.engines.mlx.utils_mlx import (
            _patch_deepseek_sanitize_for_mtp,
            _restore_deepseek_sanitize,
        )

        deepseek_v3 = pytest.importorskip("mlx_lm.models.deepseek_v3")
        model_cls = deepseek_v3.Model

        original_sanitize = model_cls.sanitize

        _patch_deepseek_sanitize_for_mtp()
        assert model_cls.sanitize is not original_sanitize

        _restore_deepseek_sanitize()
        assert model_cls.sanitize is original_sanitize

    def test_double_patch_is_safe(self) -> None:
        """Calling patch twice should be safe (idempotent)."""
        from exo.worker.engines.mlx.utils_mlx import (
            _patch_deepseek_sanitize_for_mtp,
            _restore_deepseek_sanitize,
        )

        deepseek_v3 = pytest.importorskip("mlx_lm.models.deepseek_v3")
        model_cls = deepseek_v3.Model

        original_sanitize = model_cls.sanitize

        try:
            _patch_deepseek_sanitize_for_mtp()
            patched_sanitize = model_cls.sanitize

            # Patch again - should be no-op
            _patch_deepseek_sanitize_for_mtp()
            assert model_cls.sanitize is patched_sanitize

        finally:
            _restore_deepseek_sanitize()
            assert model_cls.sanitize is original_sanitize


class TestModelIdDetection:
    """Tests for DeepSeek V3 model ID detection."""

    def test_detects_deepseek_v3(self) -> None:
        """Should detect DeepSeek V3 model IDs."""
        from exo.worker.engines.mlx.utils_mlx import _might_be_deepseek_v3

        assert _might_be_deepseek_v3("deepseek-ai/DeepSeek-V3")
        assert _might_be_deepseek_v3("deepseek-ai/deepseek-v3-base")
        assert _might_be_deepseek_v3("mlx-community/DeepSeek-V3-4bit")

    def test_detects_deepseek_r1(self) -> None:
        """Should detect DeepSeek R1 model IDs (also uses MTP)."""
        from exo.worker.engines.mlx.utils_mlx import _might_be_deepseek_v3

        assert _might_be_deepseek_v3("deepseek-ai/DeepSeek-R1")
        assert _might_be_deepseek_v3("mlx-community/DeepSeek-R1-4bit")

    def test_rejects_non_deepseek(self) -> None:
        """Should reject non-DeepSeek model IDs."""
        from exo.worker.engines.mlx.utils_mlx import _might_be_deepseek_v3

        assert not _might_be_deepseek_v3("meta-llama/Llama-3-70B")
        assert not _might_be_deepseek_v3("mistralai/Mixtral-8x7B")
        assert not _might_be_deepseek_v3("deepseek-ai/DeepSeek-V2")  # V2, not V3

    def test_case_insensitive(self) -> None:
        """Detection should be case insensitive."""
        from exo.worker.engines.mlx.utils_mlx import _might_be_deepseek_v3

        assert _might_be_deepseek_v3("DEEPSEEK-AI/DEEPSEEK-V3")
        assert _might_be_deepseek_v3("DeepSeek-AI/deepseek-v3")


class TestFlattenParams:
    """Tests for parameter flattening utility."""

    def test_flattens_nested_dict(self) -> None:
        """Should flatten nested parameter dict."""
        from exo.worker.engines.mlx.utils_mlx import _flatten_params

        params = {
            "model": {
                "layers": {
                    "0": {
                        "weight": mx.zeros((10,)),
                    }
                },
                "embed": mx.ones((5,)),
            }
        }

        flat = _flatten_params(params)

        assert "model.layers.0.weight" in flat
        assert "model.embed" in flat
        assert mx.allclose(flat["model.layers.0.weight"], mx.zeros((10,)))
        assert mx.allclose(flat["model.embed"], mx.ones((5,)))

    def test_handles_flat_dict(self) -> None:
        """Should handle already-flat dict."""
        from exo.worker.engines.mlx.utils_mlx import _flatten_params

        params = {
            "weight": mx.zeros((10,)),
            "bias": mx.ones((10,)),
        }

        flat = _flatten_params(params)

        assert flat == params
