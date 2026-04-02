#!/usr/bin/env python3
"""Loop-over-B patches for DFlash draft model (bf16 weights).

Replaces mlp_gate and mlp_up nn.Linear calls with custom bf16 LpB GEMV
for M <= 8, falling back to custom bm=8 GEMM for M > 8.

Same pattern as model_patches/qwen27b/lpb_patch.py but for bf16 nn.Linear
(no quantization scales/biases).

Usage:
    from bf16_lpb_patch import apply_bf16_lpb_patches
    apply_bf16_lpb_patches(drafter)
"""

import mlx.core as mx
import mlx.nn as nn

from exo.worker.engines.mlx.patches.qwen3_5.custom_bf16_qmv_loop_over_b import custom_bf16_qmv_loop_over_b
from exo.worker.engines.mlx.patches.qwen3_5.custom_bf16_gemm_bm8 import custom_bf16_gemm
from exo.worker.engines.mlx.patches.qwen3_5.custom_qmv_loop_over_b import custom_qmv_loop_over_b


class _QuantizedLpBLinear:
    """Drop-in replacement for QuantizedLinear using quantized LpB kernel."""

    def __init__(self, weight, scales, biases, N, K, GS):
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self._N = N
        self._K = K
        self._GS = GS

    def __call__(self, x):
        orig_shape = x.shape
        x_2d = x.reshape(-1, self._K)
        M = x_2d.shape[0]
        y = custom_qmv_loop_over_b(
            x_2d, self.weight, self.scales, self.biases,
            M, self._N, self._K, self._GS)
        return y.reshape(*orig_shape[:-1], self._N)


class _BF16LpBLinear:
    """Drop-in replacement for nn.Linear using bf16 LpB / bm8 kernels."""

    def __init__(self, weight, N, K):
        self.weight = weight
        self._N = N
        self._K = K

    def __call__(self, x):
        orig_shape = x.shape
        x_2d = x.reshape(-1, self._K)
        M = x_2d.shape[0]
        if M <= 8:
            y = custom_bf16_qmv_loop_over_b(x_2d, self.weight, M, self._N, self._K)
        else:
            y = custom_bf16_gemm(x_2d, self.weight, M, self._N, self._K, BM=8)
        return y.reshape(*orig_shape[:-1], self._N)


def apply_bf16_lpb_patches(drafter):
    """Patch DFlash drafter's mlp_gate and mlp_up with bf16 LpB."""
    patched = 0

    for layer in drafter.layers:
        # MLP gate + up (5120 -> 17408 bf16)
        for attr in ('mlp_gate', 'mlp_up'):
            proj = getattr(layer, attr)
            if isinstance(proj, nn.Linear):
                w = proj.weight
                N, K = w.shape
                setattr(layer, attr, _BF16LpBLinear(w, N, K))
                patched += 1

        # Attention o_proj (4096 -> 5120 bf16)
        o_proj = layer.self_attn.o_proj
        if isinstance(o_proj, nn.Linear):
            w = o_proj.weight
            N, K = w.shape
            layer.self_attn.o_proj = _BF16LpBLinear(w, N, K)
            patched += 1

        # MLP down_proj (17408 -> 5120 bf16)
        down = layer.mlp_down
        if isinstance(down, nn.Linear):
            w = down.weight
            N, K = w.shape
            layer.mlp_down = _BF16LpBLinear(w, N, K)
            patched += 1

        # Attention q_proj (5120 -> 4096 bf16)
        q_proj = layer.self_attn.q_proj
        if isinstance(q_proj, nn.Linear):
            w = q_proj.weight
            N, K = w.shape
            layer.self_attn.q_proj = _BF16LpBLinear(w, N, K)
            patched += 1

    # Patch fc (context projection, 25600 -> 5120 bf16)
    if isinstance(drafter.fc, nn.Linear):
        w = drafter.fc.weight
        N, K = w.shape
        drafter.fc = _BF16LpBLinear(w, N, K)
        patched += 1

    # Patch lm_head (quantized 8-bit, shared from target model)
    if drafter.lm_head is not None and isinstance(drafter.lm_head, nn.QuantizedLinear):
        lh = drafter.lm_head
        w, s, b = lh.weight, lh.scales, lh.biases
        N = w.shape[0]
        K = w.shape[1] * (32 // lh.bits)
        GS = lh.group_size

        drafter.lm_head = _QuantizedLpBLinear(w, s, b, N, K, GS)
        patched += 1

    print(f"  Patched {patched} DFlash projections with LpB")
    return patched
