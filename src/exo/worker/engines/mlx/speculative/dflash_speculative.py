#!/usr/bin/env python3
"""DFlash-specific speculative forward pass.

Combines:
1. Multi-layer hidden state capture (for DFlash draft context)
2. GDN SpeculativeArraysCache wrapping + speculative kernel swap (for rollback)

Single forward pass returns (target_hidden, pre_norm, logits) with rollback-ready caches.
"""

import mlx.core as mx


def dflash_speculative_forward(model, inputs, cache, target_layer_ids, speculative=False):
    """Run model forward, capture multi-layer hidden states + optional GDN rollback.

    Args:
        model: the loaded model
        inputs: (B, S) int token ids
        cache: cache list from make_prompt_cache()
        target_layer_ids: list of layer indices to capture hidden states from
        speculative: if True, wraps GDN caches for rollback

    Returns:
        (target_hidden, pre_norm, logits)
        - target_hidden: (B, S, n_layers * hidden) — concatenated hidden states from target layers
        - pre_norm: (B, S, hidden) — pre-RMSNorm hidden states
        - logits: (B, S, vocab) — output logits
    """
    from .mtp_module import _make_speculative_gdu

    inner = getattr(model, 'model', None) or model.language_model.model
    text_model = getattr(model, 'model', None) or model.language_model
    S = inputs.shape[1]
    do_spec = speculative and S > 1

    if hasattr(inner, 'embed_tokens'):
        hidden_states = inner.embed_tokens(inputs)
    else:
        hidden_states = inputs

    cache_list = cache if cache is not None else [None] * len(inner.layers)

    # GDN rollback setup
    gdn_spec_data = []
    if do_spec:
        from .speculative_cache import SpeculativeArraysCache
        for i, c in enumerate(cache_list):
            if c is not None and hasattr(c, 'cache') and not hasattr(c, 'offset'):
                cache_list[i] = SpeculativeArraysCache(c, S=S)
        if cache is not None:
            for i in range(len(cache)):
                cache[i] = cache_list[i]

    spec_all_states = []
    if do_spec:
        import mlx_lm.models.qwen3_5 as _qwen3_5_mod
        _orig_gdu = _qwen3_5_mod.gated_delta_update
        _qwen3_5_mod.gated_delta_update = _make_speculative_gdu(spec_all_states)

    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(hidden_states, cache_list[inner.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache_list[inner.ssm_idx])

    # Layer loop — capture multi-layer hidden + GDN pre-conv state
    target_layer_ids_set = set(target_layer_ids)
    layer_hiddens = {}

    for i, (layer, c) in enumerate(zip(inner.layers, cache_list)):
        mask = ssm_mask if layer.is_linear else fa_mask

        if do_spec and layer.is_linear:
            from .speculative_cache import SpeculativeArraysCache as _SAC
            if isinstance(c, _SAC):
                pre_conv = c[0]
                if pre_conv is None:
                    gdn = layer.linear_attn
                    pre_conv = mx.zeros(
                        (hidden_states.shape[0], gdn.conv_kernel_size - 1,
                         gdn.conv_dim), dtype=hidden_states.dtype)
                gdn_spec_data.append((hidden_states, pre_conv, c, layer))

        hidden_states = layer(hidden_states, mask=mask, cache=c)

        # Capture hidden states at target layers
        if i in target_layer_ids_set:
            layer_hiddens[i] = hidden_states

    # Restore original kernel + distribute all_states + conv_input
    if do_spec:
        _qwen3_5_mod.gated_delta_update = _orig_gdu

        gdn_idx = 0
        for layer_input, pre_conv, spec_cache, parent_layer in gdn_spec_data:
            if gdn_idx < len(spec_all_states):
                spec_cache.all_states = spec_all_states[gdn_idx]
            gdn_idx += 1

            gdn = parent_layer.linear_attn
            normed = parent_layer.input_layernorm(layer_input)
            if hasattr(gdn, 'in_proj_qkv'):
                qkv = gdn.in_proj_qkv(normed)
            else:
                q, k, v, z, b, a = gdn.fix_query_key_value_ordering(
                    gdn.in_proj_qkvz(normed), gdn.in_proj_ba(normed))
                B_dim = normed.shape[0]
                qkv = mx.concatenate(
                    [q.reshape(B_dim, S, -1), k.reshape(B_dim, S, -1),
                     v.reshape(B_dim, S, -1)], axis=-1)
            spec_cache.conv_input = mx.concatenate([pre_conv, qkv], axis=1)

    # Concatenate multi-layer hidden states
    selected = [layer_hiddens[i] for i in target_layer_ids]
    target_hidden = mx.concatenate(selected, axis=-1)

    # Final norm + lm_head
    pre_norm = hidden_states
    normed = inner.norm(hidden_states)

    if hasattr(text_model, 'lm_head'):
        logits = text_model.lm_head(normed)
    else:
        logits = inner.embed_tokens.as_linear(normed)

    return target_hidden, pre_norm, logits
