#!/usr/bin/env python3
"""Dynamic per-call loop-over-B patches for Qwen3.5-27B (bf16 or 8-bit target).

For each projection, the kernel is picked at CALL time based on the actual
M seen in the forward pass. Memoized per projection so runtime overhead is
a dict lookup after the first call at a given M.

Target-side projections during verify all see M = verify_len + 1 (uniform),
but we use dynamic picking for consistency with the drafter patch and to
support arbitrary (BS, V) without re-patching.

Patches (every layer):
  - MLP: gate_proj, up_proj, down_proj
  - GQA attn: q_proj, k_proj, v_proj, o_proj
  - GDN attn: in_proj_qkv, in_proj_z, out_proj
  - lm_head (runs during verify forward)
"""

import mlx.nn as nn

from exo.worker.engines.mlx.matmul.patches.kernel_picker import (
    pick_bf16_kernel,
    pick_int8_kernel,
)


MAX_M = 16  # Above this (prefill), fall back to the original projection


def _make_bf16_forward(original, N, K):
    cache = {}

    def forward(self_unused, x):
        M = 1
        for d in x.shape[:-1]:
            M *= d
        if M > MAX_M:
            return original(x)
        fn = cache.get(M)
        if fn is None:
            _, fn = pick_bf16_kernel(N, K, M)
            cache[M] = fn
        orig_shape = x.shape
        x_2d = x.reshape(-1, K)
        y = fn(x_2d, original.weight, M, N, K)
        return y.reshape(*orig_shape[:-1], N)

    return forward


def _make_int8_forward(original, N, K, GS):
    cache = {}

    def forward(self_unused, x):
        M = 1
        for d in x.shape[:-1]:
            M *= d
        if M > MAX_M:
            return original(x)
        fn = cache.get(M)
        if fn is None:
            _, fn = pick_int8_kernel(N, K, M)
            cache[M] = fn
        orig_shape = x.shape
        x_2d = x.reshape(-1, K)
        y = fn(x_2d, original.weight, original.scales, original.biases,
               M, N, K, GS)
        return y.reshape(*orig_shape[:-1], N)

    return forward


def _patch_proj(parent, proj_name):
    proj = getattr(parent, proj_name, None)
    if proj is None:
        return 0

    if isinstance(proj, nn.QuantizedLinear):
        N = proj.weight.shape[0]
        K = proj.weight.shape[1] * (32 // proj.bits)
        GS = proj.group_size
        forward = _make_int8_forward(proj, N, K, GS)
        setattr(parent, proj_name, type('LpBQuant', (), {
            '__call__': forward,
            'weight': proj.weight,
            'scales': proj.scales,
            'biases': proj.biases,
        })())
        return 1
    elif isinstance(proj, nn.Linear):
        N = proj.weight.shape[0]
        K = proj.weight.shape[1]
        forward = _make_bf16_forward(proj, N, K)
        setattr(parent, proj_name, type('LpBLinear', (), {
            '__call__': forward,
            'weight': proj.weight,
        })())
        return 1
    return 0


def apply_lpb_patches(model, batch_size=None, verify_len=None):
    """Patch all Qwen3.5-27B projections with dynamic LpB kernels.

    Note: batch_size / verify_len args are kept for backward compat but ignored.
    Kernel is picked at call time based on actual M.
    """
    inner = getattr(model, 'model', None) or model.language_model.model
    patched = 0

    for _li, layer in enumerate(inner.layers):
        mlp = layer.mlp
        for pn in ('gate_proj', 'up_proj', 'down_proj'):
            patched += _patch_proj(mlp, pn)

        # MoE (Qwen3NextSparseMoeBlock): dense sub-modules worth LpB-patching.
        # Silent-skip on dense 27B (attributes don't exist). Routed experts
        # (switch_mlp) use SwitchLinear which needs routing indices — left on
        # stock.
        patched += _patch_proj(mlp, 'gate')
        patched += _patch_proj(mlp, 'shared_expert_gate')
        shared = getattr(mlp, 'shared_expert', None)
        if shared is not None:
            for pn in ('gate_proj', 'up_proj', 'down_proj'):
                patched += _patch_proj(shared, pn)

        if layer.is_linear:
            attn = layer.linear_attn
            for pn in ('in_proj_qkv', 'in_proj_z', 'out_proj'):
                patched += _patch_proj(attn, pn)
        else:
            attn = layer.self_attn
            for pn in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                patched += _patch_proj(attn, pn)

    # Qwen3.5 / Qwen3.5-MoE keep lm_head on the TextModel wrapper
    # (model.language_model.lm_head), not on model or the inner Qwen3_5TextModel.
    # Check all three levels so we cover both layouts.
    for holder in (model, getattr(model, 'language_model', None), inner):
        if holder is None:
            continue
        if hasattr(holder, 'lm_head') and holder.lm_head is not None:
            patched += _patch_proj(holder, 'lm_head')
            break

    print(f"  Patched {patched} target projections with dynamic LpB")
    return patched
