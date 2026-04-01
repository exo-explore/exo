import sys

import mlx.core as mx
from mlx_lm.models.gated_delta import compute_g


def _compute_g_f32(a_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    return mx.exp(
        -mx.exp(a_log.astype(mx.float32))
        * mx.where(
            (a + dt_bias).astype(mx.float32) > 20,
            (a + dt_bias).astype(mx.float32),
            mx.log1p(mx.exp((a + dt_bias).astype(mx.float32))),
        )
    ).astype(a.dtype)


def patch_gdn_softplus() -> None:
    from mlx_lm.models import gated_delta

    gated_delta.compute_g = _compute_g_f32

    for mod in list(sys.modules.values()):
        if mod is gated_delta:
            continue
        if getattr(mod, "compute_g", None) is compute_g:
            object.__setattr__(mod, "compute_g", _compute_g_f32)
