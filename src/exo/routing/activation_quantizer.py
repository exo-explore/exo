import numpy as np
from numpy.typing import NDArray

from exo.shared.types.activation_compression import CompressedActivationPayload

INT4_MAX_LEVELS = 15.0
INT4_MIN_VAL = 0
INT4_MAX_VAL = 15
EPSILON_SCALE = 1e-8


class DynamicInt4ActivationQuantizer:
    """Vectorized per-token dynamic asymmetric INT4 activation quantizer.

    Compresses intermediate hidden states by ~75% for low-bandwidth edge networking.
    """

    @classmethod
    def quantize(
        cls,
        hidden_states: NDArray[np.floating],
        sequence_id: str = "seq_default",
        step: int = 0,
    ) -> CompressedActivationPayload:
        """Quantizes 2D float hidden states (tokens, hidden_dim) into packed INT4 nibbles."""
        array = np.asarray(hidden_states, dtype=np.float32)
        if array.ndim == 1:
            array = array[np.newaxis, :]

        num_tokens, hidden_dim = array.shape

        # Per-token min and max across the hidden dimension
        token_mins = np.min(array, axis=-1, keepdims=True)
        token_maxs = np.max(array, axis=-1, keepdims=True)

        ranges = token_maxs - token_mins
        zero_range_mask = ranges <= EPSILON_SCALE

        scales = np.where(zero_range_mask, 0.0, ranges / INT4_MAX_LEVELS)

        # Quantize and clip to [0, 15]
        # Where scale == 0, quantized value is 0
        safe_scales = np.where(zero_range_mask, 1.0, scales)
        normalized = (array - token_mins) / safe_scales
        quantized = np.where(
            zero_range_mask,
            0,
            np.clip(np.round(normalized), INT4_MIN_VAL, INT4_MAX_VAL),
        ).astype(np.uint8)

        # Pad hidden_dim to even length if necessary
        is_odd = hidden_dim % 2 != 0
        if is_odd:
            padded = np.pad(quantized, ((0, 0), (0, 1)), mode="constant", constant_values=0)
        else:
            padded = quantized

        # Pack two 4-bit nibbles into one uint8 byte
        low_nibbles = padded[:, 0::2] & 0x0F
        high_nibbles = (padded[:, 1::2] & 0x0F) << 4
        packed = low_nibbles | high_nibbles

        packed_bytes = packed.tobytes()

        return CompressedActivationPayload(
            sequence_id=sequence_id,
            step=step,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            precision="int4",
            scale_factors=scales.squeeze(axis=-1).tolist(),
            min_values=token_mins.squeeze(axis=-1).tolist(),
            packed_quantized_bytes=packed_bytes,
            is_delta_encoded=False,
        )

    @classmethod
    def dequantize(
        cls,
        payload: CompressedActivationPayload,
    ) -> NDArray[np.float32]:
        """Unpacks and restores float32 hidden states from packed INT4 payload."""
        num_tokens = payload.num_tokens
        hidden_dim = payload.hidden_dim

        packed_dim = (hidden_dim + 1) // 2
        packed_array = np.frombuffer(payload.packed_quantized_bytes, dtype=np.uint8).reshape(
            num_tokens, packed_dim
        )

        low_nibbles = packed_array & 0x0F
        high_nibbles = (packed_array >> 4) & 0x0F

        # Interleave low and high nibbles back into original dimension
        unpacked = np.empty((num_tokens, packed_dim * 2), dtype=np.float32)
        unpacked[:, 0::2] = low_nibbles
        unpacked[:, 1::2] = high_nibbles

        # Trim padding if original dimension was odd
        restored_quantized = unpacked[:, :hidden_dim]

        scales = np.array(payload.scale_factors, dtype=np.float32)[:, np.newaxis]
        min_vals = np.array(payload.min_values, dtype=np.float32)[:, np.newaxis]

        # Dequantize: X = Q * S + Min
        restored = restored_quantized * scales + min_vals
        return restored
