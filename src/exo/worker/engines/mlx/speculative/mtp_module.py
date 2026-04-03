#!/usr/bin/env python3
"""MTP (Multi-Token Prediction) module for Qwen3.5-27B.

Architecture (from llama.cpp build_mtp_head + HuggingFace config):
  1. Normalize: pre_fc_norm_hidden(hidden_state) || pre_fc_norm_embedding(embed(token))
  2. Combine: fc(concat([e_norm, h_norm])) → 5120
  3. 1 GQA decoder layer (same config as main model's full-attention layers)
     - Attention with Q/K RMSNorm + partial RoPE + output gate
  4. Final norm → shared lm_head → vocab logits

Predicts token t+2 given the main model's hidden state at position t
and the token sampled at position t+1.

Usage:
    from .mtp_module import MTPPredictor
    mtp = MTPPredictor(model, "mtp_weights.safetensors")
    # During decode:
    pre_norm, normed = mtp.get_hidden_state(input_tokens, cache)
    logits_t1 = mtp.apply_lm_head(normed)            # token t+1
    logits_t2 = mtp.predict(pre_norm, token_t1)       # token t+2
"""

import mlx.core as mx
import mlx.nn as nn


def speculative_forward(model, inputs, cache, speculative=False):
    """Run model forward pass, optionally capturing GDN per-step states for rollback.

    This is the shared core for both MTP and draft-model speculative decoding.
    It manually iterates model layers to:
    1. Wrap GDN caches in SpeculativeArraysCache when speculative=True
    2. Patch gated_delta_update to use the speculative kernel
    3. Capture per-step recurrent states and reconstruct conv_input

    Args:
        model: the loaded model (e.g. from mlx_lm.load)
        inputs: (B, S) int token ids
        cache: cache list from make_prompt_cache()
        speculative: if True, saves per-step GDN states for rollback

    Returns:
        (pre_norm, logits) — pre-RMSNorm hidden states and vocab logits
    """
    inner = getattr(model, 'model', None) or model.language_model.model
    text_model = getattr(model, 'model', None) or model.language_model
    S = inputs.shape[1]
    do_spec = speculative and S > 1

    if hasattr(inner, 'embed_tokens'):
        hidden_states = inner.embed_tokens(inputs)
    else:
        hidden_states = inputs

    cache_list = cache if cache is not None else [None] * len(inner.layers)

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

    for layer, c in zip(inner.layers, cache_list):
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

    pre_norm = hidden_states
    normed = inner.norm(hidden_states)

    if hasattr(text_model, 'lm_head'):
        logits = text_model.lm_head(normed)
    else:
        logits = inner.embed_tokens.as_linear(normed)

    return pre_norm, logits


def _make_speculative_gdu(all_states_list):
    """Create a gated_delta_update replacement that uses the speculative kernel.

    The speculative kernel is identical to the original but also outputs
    per-step recurrent states (all_states). These are appended to
    all_states_list for later assignment to SpeculativeArraysCache wrappers.

    Returns (y, state_out) — same interface as original gated_delta_update.
    """
    from .speculative_gdn_kernel import speculative_gated_delta_kernel
    from mlx_lm.models.gated_delta import compute_g

    def speculative_gated_delta_update(q, k, v, a, b, A_log, dt_bias,
                                        state=None, mask=None, use_kernel=True):
        beta = mx.sigmoid(b)
        g = compute_g(A_log, a, dt_bias)
        if state is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)
        y, state_out, all_states = speculative_gated_delta_kernel(
            q, k, v, g, beta, state, mask)
        all_states_list.append(all_states)
        return y, state_out

    return speculative_gated_delta_update


class MTPPredictor:
    """MTP draft predictor for speculative decoding.

    Wraps the MTP module weights and provides:
      - get_hidden_state(): extract pre-lm_head hidden state from main model
      - predict(): run MTP to get next-next-token logits
    """

    def __init__(self, model, mtp_weights_path, quantize=True, skip_mlp=False):
        """Load MTP weights and attach to the main model.

        Args:
            model: loaded Qwen3.5-27B model
            mtp_weights_path: path to mtp_weights.safetensors
            quantize: quantize MTP linears to 8-bit gs=64
            skip_mlp: skip MoE/MLP weights (saves ~13GB for PP mode)
        """
        self.model = model
        self._inner = getattr(model, 'model', None) or model.language_model.model
        self._text_model = getattr(model, 'model', None) or model.language_model

        # Shared components
        self.embed_tokens = self._inner.embed_tokens
        if hasattr(self._text_model, 'lm_head'):
            self.lm_head = self._text_model.lm_head
        else:
            # tie_word_embeddings case
            self.lm_head = None

        # Load MTP weights
        weights = mx.load(mtp_weights_path)

        # ---- Sanitize norm weights ----
        # CRITICAL: Qwen3.5 HuggingFace format stores ALL norm weights as (actual - 1.0).
        # mlx-lm's TextModel.sanitize() adds +1.0 back for the main model norms, but
        # MTP weights are stripped before sanitize runs. We must apply the same shift
        # to ALL 1-D norm weights in the MTP.
        #
        # Evidence: pre_fc_norm_hidden has mean=-0.17 raw → 0.83 after shift (plausible).
        # Linear projection weights (2-D) are NOT shifted.
        shifted = []
        for k in list(weights.keys()):
            if weights[k].ndim == 1:
                weights[k] = weights[k] + 1.0
                shifted.append(k)
        if shifted:
            print(f"  Sanitized {len(shifted)} norm weights (+1.0 shift)")

        # Detect pre-quantized weights (have .scales/.biases companions)
        _is_prequantized = any(k.endswith('.scales') for k in weights)

        # Infer all dimensions from weight shapes (works for any Qwen3.5 size)
        # For pre-quantized 4-bit: shape[0] = output_dims (unchanged),
        # shape[1] = input_dims * bits / 32 (packed). Unpack with * 32 / bits.
        def _dim(w, axis):
            """Get original dimension, unpacking if pre-quantized.
            Only axis 1 is packed (input_dims * bits / 32). Axis 0 is output_dims (unchanged).
            """
            d = w.shape[axis]
            if _is_prequantized and w.dtype == mx.uint32 and axis == 1:
                d = d * 32 // 4  # 4-bit packing: unpack input_dims
            return d

        fc_w = weights['mtp.fc.weight']
        hidden_size = _dim(fc_w, 0)                    # 4096 (9B) or 5120 (27B)
        fc_in = _dim(fc_w, 1)                          # 2 * hidden_size

        q_w = weights['mtp.layers.0.self_attn.q_proj.weight']
        q_out = _dim(q_w, 0)                           # num_heads * head_dim * 2 (gate)
        k_w = weights['mtp.layers.0.self_attn.k_proj.weight']
        kv_out = _dim(k_w, 0)                          # num_kv_heads * head_dim
        o_w = weights['mtp.layers.0.self_attn.o_proj.weight']
        o_in = _dim(o_w, 1)                            # num_heads * head_dim

        # Detect MoE vs dense MLP
        self.is_moe = 'mtp.layers.0.mlp.gate.weight' in weights

        if not self.is_moe:
            gate_w = weights['mtp.layers.0.mlp.gate_proj.weight']
            intermediate = gate_w.shape[0]
        else:
            intermediate = 0  # MoE experts handle this

        # head_dim from q_norm weight (always per-head)
        head_dim = weights.get('mtp.layers.0.self_attn.q_norm.weight',
                               mx.ones(256)).shape[0]
        num_heads = o_in // head_dim
        num_kv_heads = kv_out // head_dim

        print(f"  Dims: hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}, "
              f"head_dim={head_dim}, MLP={'MoE' if self.is_moe else f'dense({intermediate})'}")

        # Build layers from weights — all dimension-agnostic
        def make_linear(w, key_prefix: str = ''):
            if _is_prequantized and f'{key_prefix}.scales' in weights:
                # Pre-quantized: build QuantizedLinear and update weights via module.update()
                scales = weights[f'{key_prefix}.scales']
                biases = weights[f'{key_prefix}.biases']
                in_dims = w.shape[1] * 32 // 4  # unpack packed input_dims
                out_dims = w.shape[0]
                ql = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=64, bits=4)
                ql.update({'weight': w, 'scales': scales, 'biases': biases})
                return ql
            out_dim, in_dim = w.shape
            l = nn.Linear(in_dim, out_dim, bias=False)
            l.weight = w
            return l

        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size)
        self.pre_fc_norm_hidden.weight = weights['mtp.pre_fc_norm_hidden.weight']

        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size)
        self.pre_fc_norm_embedding.weight = weights['mtp.pre_fc_norm_embedding.weight']

        self.fc = make_linear(fc_w, 'mtp.fc')
        self.q_proj = make_linear(q_w, 'mtp.layers.0.self_attn.q_proj')
        self.k_proj = make_linear(k_w, 'mtp.layers.0.self_attn.k_proj')
        self.v_proj = make_linear(weights['mtp.layers.0.self_attn.v_proj.weight'], 'mtp.layers.0.self_attn.v_proj')
        self.o_proj = make_linear(o_w, 'mtp.layers.0.self_attn.o_proj')

        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        q_norm_key = 'mtp.layers.0.self_attn.q_norm.weight'
        k_norm_key = 'mtp.layers.0.self_attn.k_norm.weight'
        if q_norm_key in weights:
            self.q_norm.weight = weights[q_norm_key]
            self.k_norm.weight = weights[k_norm_key]

        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.input_layernorm.weight = weights['mtp.layers.0.input_layernorm.weight']

        self.post_attention_layernorm = nn.RMSNorm(hidden_size)
        self.post_attention_layernorm.weight = weights['mtp.layers.0.post_attention_layernorm.weight']

        self.skip_mlp = skip_mlp

        if self.is_moe and not skip_mlp:
            # Reuse mlx-lm's SparseMoeBlock from the target model
            moe_layer = None
            for layer in self._inner.layers:
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                    moe_layer = layer.mlp
                    break
            if moe_layer is None:
                raise RuntimeError("MTP has MoE weights but target model has no MoE layer")

            # Create a new MoE block with same class/config as target
            moe_class = type(moe_layer)
            args = getattr(self._text_model, 'args', None)
            if args is None and hasattr(self._text_model, 'model'):
                args = getattr(self._text_model.model, 'args', None)
            self.mlp = moe_class(args)

            # Load MTP MoE weights — remap HF expert names to mlx-lm SwitchLinear
            prefix = 'mtp.layers.0.mlp.'
            direct_keys = {}
            expert_weights = {}  # {proj_name: {expert_idx: weight}}
            for k, v in weights.items():
                if not k.startswith(prefix):
                    continue
                name = k[len(prefix):]
                if name.startswith('experts.'):
                    # experts.N.{gate,up,down}_proj.{weight,scales,biases}
                    parts = name.split('.')
                    idx = int(parts[1])
                    proj = parts[2]  # gate_proj, up_proj, down_proj
                    suffix = '.'.join(parts[3:])  # weight, scales, or biases
                    key = f'{proj}.{suffix}'
                    if key not in expert_weights:
                        expert_weights[key] = {}
                    expert_weights[key][idx] = v
                else:
                    direct_keys[name] = v

            # Stack individual expert weights into SwitchLinear format
            moe_weights = []
            for proj_key, idx_map in expert_weights.items():
                n_experts = max(idx_map.keys()) + 1
                stacked = mx.stack([idx_map[i] for i in range(n_experts)])
                moe_weights.append((f'switch_mlp.{proj_key}', stacked))

            for name, v in direct_keys.items():
                moe_weights.append((name, v))

            self.mlp.load_weights(moe_weights, strict=False)
            print(f"  MoE MLP: {len(moe_weights)} weight groups loaded "
                  f"({len(expert_weights)} stacked expert projections)")
        elif skip_mlp:
            print(f"  MLP skipped (skip_mlp=True)")
        else:
            self.gate_proj = make_linear(gate_w, 'mtp.layers.0.mlp.gate_proj')
            self.up_proj = make_linear(weights['mtp.layers.0.mlp.up_proj.weight'])
            self.down_proj = make_linear(weights['mtp.layers.0.mlp.down_proj.weight'])

        self.norm = nn.RMSNorm(hidden_size)
        self.norm.weight = weights['mtp.norm.weight']

        # RoPE from main model's GQA layers
        for layer in self._inner.layers:
            if not layer.is_linear:
                self.rope = layer.self_attn.rope
                break

        # GQA config
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # MTP KV cache (separate from main model)
        self.kv_cache = None

        mx.eval(self.pre_fc_norm_hidden.weight, self.pre_fc_norm_embedding.weight,
                self.fc.weight, self.input_layernorm.weight,
                self.post_attention_layernorm.weight, self.norm.weight,
                self.q_norm.weight, self.k_norm.weight)

        if quantize:
            self._quantize_linears()

        total_params = sum(w.size for w in weights.values())
        q_label = ' (pre-quantized 4-bit)' if _is_prequantized else (' (quantized 8-bit gs=64)' if quantize else ' (bf16)')
        print(f"  MTP loaded: {len(weights)} tensors, {total_params / 1e6:.1f}M params{q_label}")

    def _quantize_linears(self):
        """Quantize all MTP linear layers to 8-bit gs=64."""
        for name in ['fc', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
                      'gate_proj', 'up_proj', 'down_proj']:
            linear = getattr(self, name, None)
            if linear is None:
                continue  # MoE models don't have dense MLP projections
            linear.weight = linear.weight.astype(mx.bfloat16)
            q = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
            mx.eval(q.parameters())
            setattr(self, name, q)
        # For MoE, quantize expert weights in-place to reduce memory
        # (nn.quantize on the whole block OOMs — quantize one expert at a time)
        if self.is_moe and hasattr(self, 'mlp'):
            if hasattr(self.mlp, 'switch_mlp'):
                nn.quantize(self.mlp.switch_mlp, group_size=64, bits=8)
                mx.eval(self.mlp.switch_mlp.parameters())
            if hasattr(self.mlp, 'shared_expert'):
                nn.quantize(self.mlp.shared_expert, group_size=64, bits=8)
                mx.eval(self.mlp.shared_expert.parameters())

    def reset_cache(self):
        """Reset the MTP KV cache (call at start of generation)."""
        from mlx_lm.models.cache import KVCache
        self.kv_cache = KVCache()

    def get_hidden_state(self, inputs, cache, speculative=False):
        """Run main model and return pre-norm hidden states + logits.

        Delegates to the shared speculative_forward() function.
        """
        return speculative_forward(self.model, inputs, cache, speculative)

    def _attn_mlp(self, h):
        """Run GQA attention + MLP. Shared by predict, predict_hidden, predict_from_hidden."""
        B, S = h.shape[0], h.shape[1]

        residual = h
        h = self.input_layernorm(h)

        q_out = self.q_proj(h)
        q_out, gate = mx.split(
            q_out.reshape(B, S, self.num_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, S, -1)

        queries = self.q_norm(q_out).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            self.k_proj(h).reshape(B, S, self.num_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = self.v_proj(h).reshape(
            B, S, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if self.kv_cache is not None:
            offset = self.kv_cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys, values = self.kv_cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        mask = None
        if S > 1:
            total_kv = keys.shape[2]
            q_pos = mx.arange(S) + (total_kv - S)
            k_pos = mx.arange(total_kv)
            mask = mx.where(k_pos[None, :] <= q_pos[:, None],
                           mx.array(0, dtype=queries.dtype),
                           mx.array(-1e9, dtype=queries.dtype))

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)

        h = residual + self.o_proj(output * mx.sigmoid(gate))

        if self.skip_mlp:
            return h  # post-attention, skip FFN (lightweight mode for PP)

        residual = h
        h = self.post_attention_layernorm(h)
        if self.is_moe:
            h = residual + self.mlp(h)
        else:
            h = residual + self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))

        return h  # post-FFN, pre-norm

    def _combine(self, hidden_state, token_ids):
        """Combine hidden state + token embedding → fc input."""
        B, S = hidden_state.shape[0], hidden_state.shape[1]
        embed = self.embed_tokens(token_ids.reshape(B, S))
        h_norm = self.pre_fc_norm_hidden(hidden_state)
        e_norm = self.pre_fc_norm_embedding(embed)
        return self.fc(mx.concatenate([e_norm, h_norm], axis=-1))

    def predict(self, hidden_state, token_ids, return_hidden=False, draft_mode=False):
        """Predict next-next-token logits using MTP.

        Args:
            hidden_state: (B, S, D) bf16 — PRE-NORM hidden states
            token_ids: (B, S) or (S,) int — tokens at each position
            return_hidden: if True, also return pre-norm hidden for chaining
            draft_mode: if True, use truncated lm_head (32K vocab) for speed
        Returns:
            logits: (B, S, vocab_size) if S>1, (B, vocab_size) if S=1
            If return_hidden: (logits, hidden)
        """
        S = hidden_state.shape[1]
        h = self._combine(hidden_state, token_ids)
        pre_norm_out = self._attn_mlp(h)

        normed = self.norm(pre_norm_out)
        if draft_mode:
            logits = normed @ self.draft_lm_head_weight.T
        elif self.lm_head is not None:
            logits = self.lm_head(normed)
        else:
            logits = self.embed_tokens.as_linear(normed)

        if S == 1:
            logits = logits.squeeze(1)

        if return_hidden:
            return logits, pre_norm_out
        return logits

    def predict_hidden(self, hidden_state, token_ids):
        """Like predict() but returns only post-FFN hidden state (no lm_head)."""
        h = self._combine(hidden_state, token_ids)
        return self._attn_mlp(h)

    def predict_from_hidden(self, prev_hidden):
        """MTP step using post_norm of prev_hidden instead of token embedding.

        Replaces embed_tokens + pre_fc_norm_embedding with just norm(prev_hidden).
        This skips the lm_head → argmax → embed_tokens roundtrip.
        """
        post_norm = self.norm(prev_hidden)
        h_norm = self.pre_fc_norm_hidden(prev_hidden)
        h = self.fc(mx.concatenate([post_norm, h_norm], axis=-1))
        return self._attn_mlp(h)


def draft_tokens(mtp_pred, hidden, first_token_arr, gamma, temp, fast_lm_head=False):
    """Draft γ tokens by chaining MTP predictions — fully lazy, no mx.eval.

    The entire chain stays in the MLX computation graph. Draft token ids
    are lazy mx.arrays (argmax/categorical results), not Python ints.

    Args:
        first_token_arr: mx.array of shape (1,1) — the token to start from
    Returns: (draft_ids, draft_probs) where draft_ids[i] is a lazy mx.array
             scalar, draft_probs[i] is the full draft distribution (or None if greedy)
    """
    draft_ids = []
    draft_probs = []
    h = hidden
    tok_arr = first_token_arr

    for i in range(gamma):
        logits, h = mtp_pred.predict(h, tok_arr, return_hidden=True,
                                      draft_mode=fast_lm_head)

        if temp == 0:
            tok_arr = mx.argmax(logits, axis=-1).reshape(1, 1)
            draft_ids.append(tok_arr.reshape(-1))
            draft_probs.append(None)
        else:
            q = mx.softmax(logits / temp, axis=-1)
            tok_arr = mx.random.categorical(logits * (1.0 / temp)).reshape(1, 1)
            draft_ids.append(tok_arr.reshape(-1))
            draft_probs.append(q)

    return draft_ids, draft_probs
