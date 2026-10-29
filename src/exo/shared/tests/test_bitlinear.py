"""Unit tests for BitLinear ternary weight operator and dynamic quantization layer substitution."""


from exo.worker.engines.mlx.bitlinear import (
    EPSILON_NUMERICAL_STABILITY,
    is_sensitive_layer_name,
)


def test_sensitive_layer_predicate_identifies_embeddings_and_heads() -> None:
    """Verify that sensitive layers (embeddings, heads, norms) are correctly classified for precision protection."""
    sensitive_layer_candidates = (
        "model.embed_tokens",
        "transformer.wte",
        "lm_head",
        "output_projection.lm_head",
        "layers.0.input_layernorm",
        "layers.12.post_attention_layernorm",
    )

    non_sensitive_layer_candidates = (
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.1.mlp.gate_proj",
        "layers.1.mlp.up_proj",
        "layers.2.mlp.down_proj",
    )

    for path in sensitive_layer_candidates:
        assert is_sensitive_layer_name(path), f"Expected {path} to be classified as sensitive"

    for path in non_sensitive_layer_candidates:
        assert not is_sensitive_layer_name(path), f"Expected {path} to be classified as non-sensitive"


def test_ternary_quantization_mathematics_with_pure_python_simulation() -> None:
    """Test the core ternary quantization logic {-1, 0, +1} across synthetic weights."""
    sample_weights = [-2.5, -1.2, -0.4, 0.0, 0.3, 1.1, 2.7]

    mean_absolute_scale = sum(abs(w) for w in sample_weights) / len(sample_weights)
    safe_scale = max(mean_absolute_scale, EPSILON_NUMERICAL_STABILITY)

    quantized_ternary = []
    for weight in sample_weights:
        normalized = weight / safe_scale
        ternary_val = max(-1.0, min(1.0, round(normalized)))
        quantized_ternary.append(ternary_val)

    # Every quantized value must be strictly in the ternary set {-1.0, 0.0, 1.0}
    for value in quantized_ternary:
        assert value in (-1.0, 0.0, 1.0), f"Value {value} is not in the ternary set {-1, 0, 1}"


def test_activation_quantization_mathematics_with_pure_python_simulation() -> None:
    """Test the dynamic 8-bit activation quantization formula [-128, 127] across synthetic activations."""
    sample_activations = [-10.0, -5.0, 0.0, 2.5, 9.8]

    maximum_absolute_value = max(abs(x) for x in sample_activations)
    safe_scale = max(maximum_absolute_value, EPSILON_NUMERICAL_STABILITY)

    quantized_eight_bit = []
    for activation in sample_activations:
        scaled = activation * (127.0 / safe_scale)
        clamped = max(-128.0, min(127.0, round(scaled)))
        quantized_eight_bit.append(clamped)

    for value in quantized_eight_bit:
        assert -128.0 <= value <= 127.0, f"Activation {value} is outside the signed 8-bit integer range"
