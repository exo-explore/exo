from tinygrad.tensor import Tensor


def affine_dequantize(
    quantized: Tensor,
    scales: Tensor,
    biases: Tensor,
    group_size: int,
) -> Tensor:
    """
        Apply group-wise affine dequantization:
            out = quantized * scale + bias
        This is the same algorithm used by MLX quantization
    """

    extended_scales = scales.repeat_interleave(group_size, dim=-1)
    extended_biases = biases.repeat_interleave(group_size, dim=-1)

    return quantized * extended_scales + extended_biases
