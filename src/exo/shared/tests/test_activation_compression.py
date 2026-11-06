import numpy as np
from numpy.typing import NDArray

from exo.routing.activation_manager import ActivationCompressionManager
from exo.routing.activation_quantizer import DynamicInt4ActivationQuantizer
from exo.routing.temporal_delta import (
    TemporalDeltaCompressor,
    TemporalDeltaDecompressor,
)


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = float(np.dot(a_flat, b_flat))
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0 if norm_a == norm_b else 0.0
    return dot / (norm_a * norm_b)


def test_dynamic_int4_quantization_accuracy() -> None:
    # Typical hidden states of LLM (batch/tokens = 4, hidden_dim = 4096)
    np.random.seed(42)
    original_states = np.random.randn(4, 4096).astype(np.float32)

    payload = DynamicInt4ActivationQuantizer.quantize(
        original_states, sequence_id="seq-test-int4", step=1
    )
    restored = DynamicInt4ActivationQuantizer.dequantize(payload)

    # 1. Cosine similarity for 4-bit quantization on Gaussian distribution typically ~0.99
    cos_sim = cosine_similarity(original_states, restored)
    assert cos_sim > 0.985, f"Cosine similarity {cos_sim} too low"

    # 2. Shape must match exactly
    assert restored.shape == (4, 4096)

    # 3. Compression ratio must be approximately 25% of original FP16 (75% savings)
    assert payload.original_fp16_size_in_bytes == 4 * 4096 * 2  # 32768 bytes
    assert payload.compressed_payload_size_in_bytes < 9000  # < 9KB
    assert payload.compression_ratio < 0.28


def test_odd_hidden_dimension_padding() -> None:
    # Test handling of odd dimension length requiring padding
    original = np.random.randn(2, 513).astype(np.float32)
    payload = DynamicInt4ActivationQuantizer.quantize(original)
    restored = DynamicInt4ActivationQuantizer.dequantize(payload)

    assert restored.shape == (2, 513)
    assert cosine_similarity(original, restored) > 0.98


def test_constant_and_zero_tensors() -> None:
    # Zero tensor
    zeros = np.zeros((2, 256), dtype=np.float32)
    payload_zeros = DynamicInt4ActivationQuantizer.quantize(zeros)
    restored_zeros = DynamicInt4ActivationQuantizer.dequantize(payload_zeros)
    assert np.allclose(zeros, restored_zeros, atol=1e-5)

    # Uniform constant tensor
    constants = np.full((2, 256), 3.1415, dtype=np.float32)
    payload_const = DynamicInt4ActivationQuantizer.quantize(constants)
    restored_const = DynamicInt4ActivationQuantizer.dequantize(payload_const)
    assert np.allclose(constants, restored_const, atol=1e-5)


def test_temporal_delta_compression_multi_step() -> None:
    compressor = TemporalDeltaCompressor(sparsity_threshold_ratio=0.03)
    decompressor = TemporalDeltaDecompressor()

    np.random.seed(123)
    hidden_dim = 2048

    # Step 0: Baseline token
    state_step0 = np.random.randn(1, hidden_dim).astype(np.float32)
    payload0 = compressor.compress_step(state_step0, step=0)
    assert payload0.is_delta_encoded is False

    reconstructed0 = decompressor.decompress_step(payload0)
    assert cosine_similarity(state_step0, reconstructed0) > 0.985

    # Simulate subsequent 5 autoregressive decoding steps with high temporal correlation
    current_state = state_step0.copy()
    for step in range(1, 6):
        # Hidden state evolves slightly
        current_state = current_state + np.random.randn(1, hidden_dim).astype(np.float32) * 0.05
        payload = compressor.compress_step(current_state, step=step)
        assert payload.is_delta_encoded is True

        reconstructed = decompressor.decompress_step(payload)
        cos_sim = cosine_similarity(current_state, reconstructed)
        assert cos_sim > 0.985, f"Step {step} Cosine similarity {cos_sim} too low"


def test_activation_compression_manager_lifecycle() -> None:
    manager = ActivationCompressionManager(network_bandwidth_mbps=100.0)
    assert manager.should_compress is True

    # 1. Prefill step with 128 prompt tokens
    prefill_states = np.random.randn(128, 2048).astype(np.float32)
    prefill_payload = manager.process_outbound_activations(prefill_states, step=0)
    assert prefill_payload.is_delta_encoded is False
    assert prefill_payload.num_tokens == 128

    restored_prefill = manager.process_inbound_activations(prefill_payload)
    assert restored_prefill.shape == (128, 2048)

    # 2. Decode steps with 1 token each
    decode_step1 = np.random.randn(1, 2048).astype(np.float32)
    decode_payload1 = manager.process_outbound_activations(decode_step1, step=1)
    restored_decode1 = manager.process_inbound_activations(decode_payload1)
    assert restored_decode1.shape == (1, 2048)

    decode_step2 = decode_step1 + np.random.randn(1, 2048).astype(np.float32) * 0.03
    decode_payload2 = manager.process_outbound_activations(decode_step2, step=2)
    assert decode_payload2.is_delta_encoded is True
    restored_decode2 = manager.process_inbound_activations(decode_payload2)
    assert cosine_similarity(decode_step2, restored_decode2) > 0.985

    # 3. Verify bandwidth reduction percentage
    assert manager.total_bytes_saved > 0
    assert manager.overall_saving_percentage > 70.0
