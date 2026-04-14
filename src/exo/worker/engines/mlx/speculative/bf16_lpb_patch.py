#!/usr/bin/env python3
"""Dynamic per-call loop-over-B patches for the DFlash draft model.

For each patched projection, the kernel is picked at CALL time based on
the actual M seen in the forward pass (via matmul/patches/kernel_picker).
Each projection memoizes its kernel cache, so after warmup the pick is
O(1) dict lookup.

This is correct for any (block_size, verify_len) combo because each
projection sees a different M (e.g. k_proj sees M = S_ctx + block_size,
mlp_gate sees M = block_size, fc sees M = S_ctx, lm_head sees M = block_size - 1).
"""

import os

import mlx.nn as nn

from exo.worker.engines.mlx.matmul.patches.kernel_picker import (
    pick_bf16_kernel,
    pick_int8_kernel,
)


MAX_M = 16  # Above this (prefill), fall back to the original projection


class _BF16LpBLinear:
    """nn.Linear drop-in with dynamic per-call kernel selection."""

    def __init__(self, original, N, K):
        self._orig = original  # kept for prefill fallback and weight access
        self.weight = original.weight
        self._N = N
        self._K = K
        self._cache = {}       # M → fn
        self._name_log = {}    # M → kernel name (debug)

    def __call__(self, x):
        M = 1
        for d in x.shape[:-1]:
            M *= d
        if M > MAX_M:
            return self._orig(x)
        fn = self._cache.get(M)
        if fn is None:
            name, fn = pick_bf16_kernel(self._N, self._K, M)
            self._cache[M] = fn
            self._name_log[M] = name
        orig_shape = x.shape
        x_2d = x.reshape(-1, self._K)
        y = fn(x_2d, self.weight, M, self._N, self._K)
        return y.reshape(*orig_shape[:-1], self._N)


class _QuantizedLpBLinear:
    """nn.QuantizedLinear drop-in with dynamic per-call kernel selection."""

    def __init__(self, original, N, K, GS):
        self._orig = original
        self.weight = original.weight
        self.scales = original.scales
        self.biases = original.biases
        self._N = N
        self._K = K
        self._GS = GS
        self._cache = {}
        self._name_log = {}

    def __call__(self, x):
        M = 1
        for d in x.shape[:-1]:
            M *= d
        if M > MAX_M:
            return self._orig(x)
        fn = self._cache.get(M)
        if fn is None:
            name, fn = pick_int8_kernel(self._N, self._K, M)
            self._cache[M] = fn
            self._name_log[M] = name
        orig_shape = x.shape
        x_2d = x.reshape(-1, self._K)
        y = fn(x_2d, self.weight, self.scales, self.biases, M, self._N, self._K, self._GS)
        return y.reshape(*orig_shape[:-1], self._N)


def _wrap(proj):
    if isinstance(proj, nn.QuantizedLinear):
        N = proj.weight.shape[0]
        K = proj.weight.shape[1] * (32 // proj.bits)
        GS = proj.group_size
        return _QuantizedLpBLinear(proj, N, K, GS)
    elif isinstance(proj, nn.Linear):
        N = proj.weight.shape[0]
        K = proj.weight.shape[1]
        return _BF16LpBLinear(proj, N, K)
    return proj


def apply_bf16_lpb_patches(drafter):
    """Patch DFlash drafter's projections with dynamic LpB kernels.

    Set env DFLASH_LPB_ONLY to a comma-separated list of projection names
    to patch only those (for bisecting which patch breaks things).
    Names: mlp_gate, mlp_up, mlp_down, q_proj, k_proj, v_proj, o_proj, fc, lm_head
    """
    only_str = os.environ.get("DFLASH_LPB_ONLY", "")
    only = set(only_str.split(",")) if only_str else None
    if only:
        print(f"  DFLASH_LPB_ONLY={only_str}")

    def should_patch(name):
        return only is None or name in only

    patched = 0

    for layer in drafter.layers:
        for attr in ('mlp_gate', 'mlp_up', 'mlp_down'):
            if not should_patch(attr):
                continue
            proj = getattr(layer, attr, None)
            if proj is not None and isinstance(proj, nn.Linear):
                setattr(layer, attr, _wrap(proj))
                patched += 1

        for attr in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            if not should_patch(attr):
                continue
            proj = getattr(layer.self_attn, attr, None)
            if proj is not None and isinstance(proj, nn.Linear):
                setattr(layer.self_attn, attr, _wrap(proj))
                patched += 1

    if should_patch('fc') and hasattr(drafter, 'fc') and isinstance(drafter.fc, nn.Linear):
        drafter.fc = _wrap(drafter.fc)
        patched += 1

    if should_patch('lm_head') and drafter.lm_head is not None and isinstance(
        drafter.lm_head, (nn.Linear, nn.QuantizedLinear)
    ):
        drafter.lm_head = _wrap(drafter.lm_head)
        patched += 1

    print(f"  Patched {patched} DFlash drafter projections with dynamic LpB")
    return patched
