import math

import mlx.core as mx
from mlx_lm.models import rope_utils

_original_YarnRoPE_init = rope_utils.YarnRoPE.__init__  # noqa: N816
_original_initialize_rope = rope_utils.initialize_rope


def _patched_yarn_init(
    self: rope_utils.YarnRoPE,
    dims: int,
    traditional: bool = False,
    max_position_embeddings: int = 2048,
    base: float = 10000,
    scaling_factor: float = 1.0,
    original_max_position_embeddings: int = 4096,
    beta_fast: float = 32,
    beta_slow: float = 1,
    mscale: float = 1,
    mscale_all_dim: float = 0,
    truncate: bool = True,
) -> None:
    """Patch mlx_lm's YarnRoPE to match vLLM's inverse-frequency blending formula for compatability."""

    super(rope_utils.YarnRoPE, self).__init__()

    def yarn_find_correction_dim(num_rotations: float) -> float:
        return (
            dims
            * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

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

    self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
        scaling_factor, mscale_all_dim
    )
    pos_freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
    low, high = yarn_find_correction_range()
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_mask)
        + inv_freq_extrapolation * inv_freq_mask
    )
    self._freqs = 1.0 / inv_freq
    self.dims = dims
    self.traditional = traditional


def _patched_initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: dict[str, str | int | float | bool] | None = None,
    max_position_embeddings: int | None = None,
) -> object:
    rope_type = "default"
    if scaling_config is not None:
        rope_type = str(
            scaling_config.get("type") or scaling_config.get("rope_type", "default")
        )

    # All the yarn rope types supported in mlx lm
    if rope_type in ("yarn", "deepseek_yarn"):
        assert scaling_config is not None
        cfg = scaling_config

        def _float(key: str, default: float) -> float:
            v = cfg.get(key)
            return float(v) if v is not None else default

        def _int(key: str, default: int) -> int:
            v = cfg.get(key)
            return int(v) if v is not None else default

        return rope_utils.YarnRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings or 2048,
            traditional=traditional,
            scaling_factor=_float("factor", 1.0),
            base=base,
            original_max_position_embeddings=_int(
                "original_max_position_embeddings", 4096
            ),
            beta_fast=_float("beta_fast", 32),
            beta_slow=_float("beta_slow", 1),
            mscale=_float("mscale", 1),
            mscale_all_dim=_float("mscale_all_dim", 0),
        )

    return _original_initialize_rope(
        dims, base, traditional, scaling_config, max_position_embeddings
    )


def patch_yarn_rope() -> None:
    rope_utils.YarnRoPE.__init__ = _patched_yarn_init
    rope_utils.initialize_rope = _patched_initialize_rope
