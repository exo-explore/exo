"""Patch mlx_lm's GDN gated_delta_update to match vLLM's float32 precision.

vLLM computes both softplus (gating) and sigmoid (beta) in float32.
mlx_lm computes them in bfloat16. The precision difference compounds
through the SSM recurrence over thousands of tokens.
"""

import sys
from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def _compute_g_f32(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    return mx.exp(
        -mx.exp(A_log.astype(mx.float32))
        * nn.softplus((a + dt_bias).astype(mx.float32))
    )


def patch_gdn_softplus() -> None:
    from mlx_lm.models import gated_delta

    orig_update = gated_delta.gated_delta_update
    orig_ops = gated_delta.gated_delta_ops
    orig_kernel = gated_delta.gated_delta_kernel

    def patched_gated_delta_update(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        a: mx.array,
        b: mx.array,
        A_log: mx.array,
        dt_bias: mx.array,
        state: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        use_kernel: bool = True,
    ) -> Tuple[mx.array, mx.array]:
        beta = mx.sigmoid(b.astype(mx.float32)).astype(b.dtype)
        g = _compute_g_f32(A_log, a, dt_bias)
        if state is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)

        return orig_ops(q, k, v, g, beta, state, mask)

    gated_delta.gated_delta_update = patched_gated_delta_update

    for mod in list(sys.modules.values()):
        if mod is None or mod is gated_delta:
            continue
        if getattr(mod, "gated_delta_update", None) is orig_update:
            mod.gated_delta_update = patched_gated_delta_update
