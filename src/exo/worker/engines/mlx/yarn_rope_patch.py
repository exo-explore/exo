"""Patch mlx_lm's YarnRoPE to match vLLM's inverse-frequency blending formula.

mlx_lm's YarnRoPE uses a harmonic blend of frequencies. vLLM uses a linear blend
of inverse frequencies. These produce different rotation angles, causing KV cache
mismatch in disaggregated prefill. This patch replaces the frequency computation
to match vLLM exactly, including support for the `truncate` parameter.
"""

import math

import mlx.core as mx
from mlx_lm.models import rope_utils


_original_YarnRoPE_init = rope_utils.YarnRoPE.__init__


def _patched_yarn_init(
    self,  # type: ignore
    dims,  # type: ignore
    traditional=False,
    max_position_embeddings=2048,
    base=10000,
    scaling_factor=1.0,
    original_max_position_embeddings=4096,
    beta_fast=32,
    beta_slow=1,
    mscale=1,
    mscale_all_dim=0,
    truncate=True,
) -> None:
    super(rope_utils.YarnRoPE, self).__init__()

    def yarn_find_correction_dim(num_rotations: float) -> float:
        return (dims * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def yarn_find_correction_range() -> tuple[float, float]:
        low: float = yarn_find_correction_dim(beta_fast)
        high: float = yarn_find_correction_dim(beta_slow)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dims - 1)

    def yarn_get_mscale(scale: float = 1, ms: float = 1) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * ms * math.log(scale) + 1.0

    def yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> mx.array:
        if min_val == max_val:
            max_val += 0.001
        linear_func = (mx.arange(dim, dtype=mx.float32) - min_val) / (max_val - min_val)
        return mx.clip(linear_func, 0, 1)

    self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim)
    pos_freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
    low, high = yarn_find_correction_range()
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    self._freqs = 1.0 / inv_freq
    self.dims = dims
    self.traditional = traditional


def _patched_initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: dict | None = None,
    max_position_embeddings: int | None = None,
) -> object:  # type: ignore
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")
    else:
        rope_type = "default"

    if rope_type in ("yarn", "deepseek_yarn", "telechat3-yarn"):
        scaling_factor = scaling_config["factor"]  # type: ignore
        rope_kwargs = {
            key: scaling_config[key]  # type: ignore
            for key in ["original_max_position_embeddings", "beta_fast", "beta_slow", "mscale", "mscale_all_dim", "truncate"]
            if key in scaling_config  # type: ignore
        }
        return rope_utils.YarnRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            scaling_factor=scaling_factor,
            base=base,
            **rope_kwargs,
        )

    return _original_initialize_rope(dims, base, traditional, scaling_config, max_position_embeddings)


_original_initialize_rope = rope_utils.initialize_rope


def patch_yarn_rope() -> None:
    rope_utils.YarnRoPE.__init__ = _patched_yarn_init  # type: ignore
    rope_utils.initialize_rope = _patched_initialize_rope  # type: ignore
