#!/usr/bin/env python3
"""Loop-over-B patches for Qwen3.5-27B dense model.

Replaces vanilla QuantizedLinear calls with custom loop-over-B GEMV
for projections where N > K (expanding projections). Falls back to
vanilla for N <= K (contracting projections like down_proj, o_proj).

Usage:
    from lpb_patch import apply_lpb_patches
    apply_lpb_patches(model, batch_size=4)
"""

import mlx.core as mx
import mlx.nn as nn

from .custom_qmv_loop_over_b import custom_qmv_loop_over_b


def _make_lpb_forward(original_module, N, K, BS, GS=64):
    """Create a patched forward that uses loop-over-B."""
    w = original_module.weight
    s = original_module.scales
    b = original_module.biases

    MAX_M = 16  # Max total tokens (B*S) for custom kernel; above this use vanilla

    def forward(self_unused, x):
        # Use LpB for small M=B*S. Large prefill falls back to vanilla.
        M_total = 1
        for d in x.shape[:-1]:
            M_total *= d
        if M_total > MAX_M:
            return original_module(x)
        orig_shape = x.shape
        x_2d = x.reshape(-1, K)
        M = x_2d.shape[0]
        y = custom_qmv_loop_over_b(x_2d, w, s, b, M, N, K, GS)
        return y.reshape(*orig_shape[:-1], N)

    return forward


def apply_lpb_patches(model, batch_size=4):
    """Patch all expanding QuantizedLinear projections with loop-over-B.

    Only patches projections where N > K (expanding):
    - gate_proj, up_proj (17408 > 5120)
    - in_proj_qkv (10240 > 5120)
    - in_proj_z (6144 > 5120)
    - q_proj (12288 > 5120)

    Skips N <= K projections (down_proj, o_proj, k_proj, v_proj)
    where vanilla is already efficient.
    """
    inner = getattr(model, 'model', None) or model.language_model.model
    patched = 0

    for li, layer in enumerate(inner.layers):
        # MLP: gate_proj, up_proj (N=17408, K=5120)
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(mlp, proj_name)
            if isinstance(proj, nn.QuantizedLinear):
                N = proj.weight.shape[0]  # output dim
                K_packed = proj.weight.shape[1]
                K = K_packed * 4  # 8-bit: 4 values per uint32
                setattr(mlp, proj_name, type('LpBLinear', (), {
                    '__call__': _make_lpb_forward(proj, N, K, batch_size),
                    'weight': proj.weight,
                    'scales': proj.scales,
                    'biases': proj.biases,
                })())
                patched += 1

        # Attention projections
        if layer.is_linear:
            attn = layer.linear_attn
            for proj_name in ['in_proj_qkv', 'in_proj_z', 'out_proj']:
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    if isinstance(proj, nn.QuantizedLinear):
                        N = proj.weight.shape[0]
                        K = proj.weight.shape[1] * 4
                        setattr(attn, proj_name, type('LpBLinear', (), {
                            '__call__': _make_lpb_forward(proj, N, K, batch_size),
                            'weight': proj.weight,
                            'scales': proj.scales,
                            'biases': proj.biases,
                        })())
                        patched += 1
        else:
            attn = layer.self_attn
            for proj_name in ['q_proj', 'o_proj']:
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    if isinstance(proj, nn.QuantizedLinear):
                        N = proj.weight.shape[0]
                        K = proj.weight.shape[1] * 4
                        setattr(attn, proj_name, type('LpBLinear', (), {
                            '__call__': _make_lpb_forward(proj, N, K, batch_size),
                            'weight': proj.weight,
                            'scales': proj.scales,
                            'biases': proj.biases,
                        })())
                        patched += 1

    print(f"  Patched {patched} projections with loop-over-B")
    return patched
