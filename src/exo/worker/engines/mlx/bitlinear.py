"""BitLinear layer implementation for 1.58-bit ternary weight models on MLX.

This module provides the fundamental linear transformation layer for BitNet b1.58 architectures.
Weights are constrained to ternary values {-1, 0, +1}, and activations are quantized to 8-bit integers.
This completely removes floating-point matrix multiplications, replacing them with integer addition and subtraction.
"""

import math
from typing import final

try:
    import mlx.core as mx
    from mlx import nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # Mock base class when running in environments without the MLX dependency
    class _MockModule:
        pass
    nn = type("MockNN", (), {"Module": _MockModule})()  # type: ignore[assignment]
    mx = None  # type: ignore[assignment]


EPSILON_NUMERICAL_STABILITY: float = 1e-5


def quantize_activations_to_eight_bit(
    activations: "mx.array",
) -> tuple["mx.array", "mx.array"]:
    """Quantize activations to 8-bit signed integers [-128, 127] using per-token absmax scaling.

    Args:
        activations: Input activation tensor of arbitrary leading dimensions and trailing feature dimension.

    Returns:
        A tuple of (quantized_activations, activation_scale_factors).
    """
    assert HAS_MLX and mx is not None, "MLX runtime is required for quantize_activations_to_eight_bit"

    absolute_maximum_values = mx.max(mx.abs(activations), axis=-1, keepdims=True)
    scale_factors = mx.maximum(absolute_maximum_values, EPSILON_NUMERICAL_STABILITY)

    # Quantize to signed 8-bit range [-128, 127]
    scaled_activations = activations * (127.0 / scale_factors)
    quantized_activations = mx.clip(mx.round(scaled_activations), -128.0, 127.0)

    return quantized_activations, scale_factors


def quantize_weights_to_ternary(
    weights: "mx.array",
) -> tuple["mx.array", "mx.array"]:
    """Quantize continuous weight matrix to ternary values {-1, 0, +1} using mean absolute scaling.

    Args:
        weights: Unquantized floating-point weight matrix of shape (output_features, input_features).

    Returns:
        A tuple of (ternary_weights, weight_scale_factor).
    """
    assert HAS_MLX and mx is not None, "MLX runtime is required for quantize_weights_to_ternary"

    mean_absolute_scale = mx.mean(mx.abs(weights))
    safe_scale_factor = mx.maximum(mean_absolute_scale, EPSILON_NUMERICAL_STABILITY)

    # Scale and round weights into ternary set {-1, 0, +1}
    normalized_weights = weights / safe_scale_factor
    ternary_weights = mx.clip(mx.round(normalized_weights), -1.0, 1.0)

    return ternary_weights, safe_scale_factor


@final
class BitLinearLayer(nn.Module):
    """Linear layer computing with 1.58-bit ternary weights and 8-bit quantized activations.

    Standard linear transformations compute Y = X @ W.T with full-precision floating-point arithmetic.
    BitLinear converts weights to {-1, 0, +1} and activations to int8. Matrix multiplication thus
    reduces to integer accumulation (addition and subtraction), drastically cutting memory bandwidth
    and power draw on consumer devices.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        has_bias: bool = False,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.has_bias = has_bias

        if HAS_MLX and mx is not None:
            # Initialize weights with standard normal distribution scaled by fan-in
            scale = 1.0 / math.sqrt(input_features)
            initial_weights = mx.random.normal(
                shape=(output_features, input_features)
            ) * scale
            self.weight = initial_weights
            self.weight_scale = mx.array(1.0)

            if has_bias:
                self.bias = mx.zeros((output_features,))
            else:
                self.bias = None
        else:
            self.weight = None
            self.weight_scale = None
            self.bias = None

    def pack_and_freeze_weights(self) -> None:
        """Convert current floating-point weights into ternary representation and compute static scale."""
        if not HAS_MLX or mx is None or self.weight is None:
            return

        ternary_weights, safe_scale = quantize_weights_to_ternary(self.weight)
        self.weight = ternary_weights
        self.weight_scale = safe_scale

    def __call__(self, input_tensor: "mx.array") -> "mx.array":
        """Execute linear projection with 8-bit activations and ternary weights.

        Args:
            input_tensor: Input activations of shape (..., input_features).

        Returns:
            Output activations of shape (..., output_features).
        """
        assert HAS_MLX and mx is not None, "MLX runtime is required to execute BitLinearLayer"
        assert self.weight is not None, "Layer weights must be initialized before execution"

        # Quantize dynamic activations to 8-bit signed range
        quantized_activations, activation_scales = quantize_activations_to_eight_bit(input_tensor)

        # Integer matrix multiplication (adds and subtracts without floating point multiplications)
        integer_accumulated_output = mx.matmul(
            quantized_activations,
            self.weight.transpose(),
        )

        # Rescale the accumulated integers back to floating point output representation
        combined_scale = (activation_scales * self.weight_scale) / 127.0
        projected_output = integer_accumulated_output * combined_scale

        if self.bias is not None:
            projected_output = projected_output + self.bias

        return projected_output


def is_sensitive_layer_name(layer_path: str) -> bool:
    """Determine whether a sub-layer is sensitive and should retain higher precision.

    Embeddings and language model prediction heads carry concentrated semantic distributions,
    which quickly deteriorate if aggressively forced into ternary weights.
    """
    normalized_path = layer_path.lower()
    sensitive_substrings = ("embed", "lm_head", "wte", "wpe", "norm")
    return any(substring in normalized_path for substring in sensitive_substrings)


def apply_bitlinear_layers(
    model: nn.Module,
    preserve_sensitive_layers: bool = True,
) -> nn.Module:
    """Recursively replace eligible standard Linear layers with BitLinearLayer in an MLX model.

    Args:
        model: MLX neural network module instance.
        preserve_sensitive_layers: Whether to protect vocabulary embeddings and prediction heads.

    Returns:
        The updated model with BitLinear layers installed.
    """
    if not HAS_MLX or mx is None:
        return model

    def _replace_submodules(module: nn.Module, prefix: str = "") -> None:
        for attribute_name, child in list(module.__dict__.items()):
            current_path = f"{prefix}.{attribute_name}" if prefix else attribute_name
            if isinstance(child, nn.Linear):
                if preserve_sensitive_layers and is_sensitive_layer_name(current_path):
                    continue

                output_features, input_features = child.weight.shape
                has_bias = hasattr(child, "bias") and child.bias is not None

                bitlinear_instance = BitLinearLayer(
                    input_features=input_features,
                    output_features=output_features,
                    has_bias=has_bias,
                )
                bitlinear_instance.weight = child.weight
                if has_bias:
                    bitlinear_instance.bias = child.bias

                bitlinear_instance.pack_and_freeze_weights()
                setattr(module, attribute_name, bitlinear_instance)

            elif isinstance(child, nn.Module):
                _replace_submodules(child, current_path)

    _replace_submodules(model)
    return model
