from typing import Tuple

from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


def compute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float = 10000.0,
) -> Tuple[Tensor, Tensor]:
    dim_pairs = head_dim // 2
    freq_exponents = Tensor.arange(0, dim_pairs, dtype=dtypes.float32) * 2.0 / head_dim  # pyright: ignore[reportUnknownMemberType]
    inv_freq = 1.0 / (rope_theta ** freq_exponents)

    positions = Tensor.arange(0, max_seq_len, dtype=dtypes.float32)  # pyright: ignore[reportUnknownMemberType]
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)

    cos_freqs = angles.cos()
    sin_freqs = angles.sin()

    return cos_freqs, sin_freqs # Each: (max_seq_len, head_dim // 2)

def apply_rope(
    x: Tensor,
    cos_freqs: Tensor,
    sin_freqs: Tensor,
    position_offset: int | Tensor = 0,
) -> Tensor:
    seq_len = x.shape[2]

    cos = cos_freqs
    sin = sin_freqs

    if isinstance(position_offset, Tensor):
        # position_offset is shape (1,) — index into full tables.
        # cos_freqs is (max_seq_len, dim/2), result: (1, dim/2) → (1, 1, 1, dim/2)
        cos = cos_freqs[position_offset].reshape(1, 1, seq_len, -1)  # pyright: ignore[reportUnknownMemberType]
        sin = sin_freqs[position_offset].reshape(1, 1, seq_len, -1)  # pyright: ignore[reportUnknownMemberType]
    else:
        cos = cos_freqs[position_offset:position_offset + seq_len].reshape(1, 1, seq_len, -1)  # pyright: ignore[reportUnknownMemberType]
        sin = sin_freqs[position_offset:position_offset + seq_len].reshape(1, 1, seq_len, -1)  # pyright: ignore[reportUnknownMemberType]

    half = x.shape[-1] // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    return out1.cat(out2, dim=-1)
