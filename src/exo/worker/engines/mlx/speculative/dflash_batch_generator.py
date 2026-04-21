#!/usr/bin/env python3
"""DFlash Speculative Decoding integrated with mlx_lm's BatchGenerator.

Two prefill approaches (PREFILL_MODE env var):
  "capture" (default): Use _CapturingLayer to intercept hidden states during
                       super()._next() prefill. No extra forward pass.
  "direct":           Run dflash_speculative_forward for prefill ourselves,
                       bypassing super()._next() for the first step.
"""

import time

import mlx.core as mx
from mlx_lm.generate import BatchGenerator

from .dflash_module import DFlashDrafter
from .dflash_speculative import dflash_speculative_forward


class DFlashBatchGenerator(BatchGenerator):
    """BatchGenerator with DFlash speculative decoding for BS=1."""

    def __init__(
        self,
        model,
        drafter: DFlashDrafter,
        verify_len: int = 5,
        block_size: int = 6,
        temp: float = 0.0,
        alpha: float = 1.0,
        prefill_mode: str = "capture",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.drafter = drafter
        self.verify_len = verify_len
        self.temp = temp
        self.alpha = alpha
        self.prefill_mode = prefill_mode

        drafter.block_size = block_size

        self._token_buffer = {}
        self._last_target_hidden = {}
        self._draft_position = {}
        self._prefilled = set()
        self._request_temp = {}
        self._captured = {}

        if prefill_mode == "capture":
            self._setup_hidden_capture()

    def _setup_hidden_capture(self):
        """Replace target layers with wrappers that capture hidden states."""
        inner = getattr(self.model, 'model', None) or self.model.language_model.model
        target_ids = set(self.drafter.target_layer_ids)
        captured = self._captured

        class _CapturingLayer:
            def __init__(self, orig, layer_idx):
                self._orig = orig
                self._layer_idx = layer_idx

            def __call__(self, *args, **kwargs):
                out = self._orig(*args, **kwargs)
                if 'layer_hiddens' not in captured:
                    captured['layer_hiddens'] = {}
                captured['layer_hiddens'][self._layer_idx] = out
                if out.shape[1] > 1:
                    if 'prefill_hiddens' not in captured:
                        captured['prefill_hiddens'] = {}
                    captured['prefill_hiddens'][self._layer_idx] = out
                return out

            def __getattr__(self, name):
                return getattr(self._orig, name)

        for i in range(len(inner.layers)):
            if i in target_ids:
                inner.layers[i] = _CapturingLayer(inner.layers[i], i)

    def _get_captured_target_hidden(self, key='layer_hiddens'):
        hiddens = self._captured.get(key)
        if hiddens is None:
            return None
        selected = [hiddens[i] for i in self.drafter.target_layer_ids if i in hiddens]
        if len(selected) != len(self.drafter.target_layer_ids):
            return None
        self._captured[key] = {}
        return mx.concatenate(selected, axis=-1)

    def _next(self):
        batch = self.active_batch

        # Yield buffered tokens first
        if batch is not None and len(batch) == 1:
            uid = batch.uids[0]
            if uid in self._token_buffer and self._token_buffer[uid]:
                return self._yield_buffered(batch, uid)

        # BS=1 speculative path
        if (batch is not None
                and len(batch) == 1
                and self.verify_len > 0
                and len(self.unprocessed_prompts) == 0):
            uid = batch.uids[0]
            if uid not in self._prefilled:
                if self.prefill_mode == "direct":
                    return self._first_step_direct(batch, uid)
                else:
                    return self._first_step_capture(batch, uid)
            return self._speculative_next()

        # Standard path (BS>1 or no batch)
        return super()._next()

    # ── Approach 1: "capture" ──

    def _first_step_capture(self, batch, uid):
        """Use prefill hidden states captured by _CapturingLayer during super()._next()."""
        target_hidden = self._get_captured_target_hidden(key='prefill_hiddens')
        if target_hidden is None:
            target_hidden = self._get_captured_target_hidden(key='layer_hiddens')
        if target_hidden is not None:
            mx.eval(target_hidden)
            self._last_target_hidden[uid] = target_hidden

        for c in batch.cache:
            if hasattr(c, 'offset'):
                off = c.offset
                self._draft_position[uid] = off.item() if hasattr(off, 'item') else int(off)
                break
        self.drafter.reset_draft_cache()
        self._prefilled.add(uid)
        return self._speculative_next()

    # ── Approach 2: "direct" ──

    def _first_step_direct(self, batch, uid):
        """Run dflash_speculative_forward ourselves for prefill, then first speculative cycle."""
        # batch.y has the first token from super()._next()'s prefill
        # But we need target hidden from that prefill. Run dflash_speculative_forward
        # on the prompt tokens to get them.
        prompt_toks = batch.tokens[0]  # full prompt token history
        mx.eval(prompt_toks)

        # Run target model forward on prompt to capture multi-layer hidden
        target_hidden, _, logits = dflash_speculative_forward(
            self.model, prompt_toks.reshape(1, -1), batch.cache,
            self.drafter.target_layer_ids, speculative=False)
        mx.eval(target_hidden, logits)

        self._last_target_hidden[uid] = target_hidden

        # First token from logits
        first_token = mx.argmax(logits[0, -1], axis=-1).item()
        batch.y = mx.array([first_token])

        for c in batch.cache:
            if hasattr(c, 'offset'):
                off = c.offset
                self._draft_position[uid] = off.item() if hasattr(off, 'item') else int(off)
                break
        self.drafter.reset_draft_cache()
        self._prefilled.add(uid)
        return self._speculative_next()

    # ── Core speculative cycle ──

    def _speculative_next(self):
        tic = time.perf_counter()
        batch = self.active_batch
        uid = batch.uids[0]
        y = batch.y
        y_val = y[0].item()
        y_logprobs = batch.logprobs[0]

        batch.tokens[0] = mx.concatenate((batch.tokens[0], y[0:1]))

        last_target_hidden = self._last_target_hidden.get(uid)
        if last_target_hidden is None:
            return super()._next()

        bs = self.drafter.block_size
        verify_len = self.verify_len
        temp = self._request_temp.get(uid, self.temp)
        alpha = self.alpha
        start = self._draft_position[uid]

        # 1. Draft
        block_ids = mx.full((1, bs), self.drafter.mask_token_id, dtype=mx.int32)
        block_ids[:, 0] = y_val
        draft_logits = self.drafter.draft(last_target_hidden, block_ids, start)
        self.drafter.crop_draft_cache(start)

        # 2. Sample
        if temp == 0:
            all_drafts_arr = mx.argmax(draft_logits, axis=-1).squeeze(0)
            mx.eval(all_drafts_arr)
            all_drafts = all_drafts_arr.tolist()
            draft_probs = None
        else:
            all_drafts = []
            draft_probs = []
            for i in range(bs - 1):
                p = mx.softmax(draft_logits[0, i] / temp, axis=-1)
                tok = mx.random.categorical(mx.log(p)).item()
                all_drafts.append(tok)
                draft_probs.append(p)

        drafts = all_drafts[:verify_len]

        # 3. Verify
        verify_input = mx.array([[y_val] + drafts])
        target_hidden, _, verify_logits = dflash_speculative_forward(
            self.model, verify_input, batch.cache,
            self.drafter.target_layer_ids, speculative=True)

        # Build acceptance lazily (no logprobs — skip expensive logsumexp on 248K vocab)
        if temp == 0:
            target_tokens = mx.argmax(verify_logits[:, :verify_len, :], axis=-1)
            draft_arr = mx.array([drafts])
            matches = mx.equal(target_tokens, draft_arr).squeeze(0)
            all_next = mx.argmax(verify_logits[0], axis=-1)
            mx.async_eval(matches, all_next, target_hidden)
        else:
            accept_ratios = []
            for i in range(verify_len):
                p = mx.softmax(verify_logits[0, i] / temp, axis=-1)
                q = draft_probs[i]
                p_di = p[drafts[i]]
                q_di = mx.maximum(q[drafts[i]], mx.array(1e-10))
                ratio = p_di / q_di
                accept_ratios.append(mx.minimum(ratio ** alpha, mx.array(1.0)))
            uniforms = mx.random.uniform(shape=(verify_len,))
            corrections = []
            for i in range(verify_len):
                p = mx.softmax(verify_logits[0, i] / temp, axis=-1)
                q = draft_probs[i]
                residual = mx.maximum(p - q, 0.0)
                corrections.append(mx.random.categorical(mx.log(residual + 1e-10)))
            bonus_token = mx.random.categorical(verify_logits[0, verify_len] * (1.0 / temp))
            mx.async_eval(accept_ratios, uniforms, corrections, bonus_token, target_hidden)

        # 4. Accept
        n_accepted = 0
        for i in range(verify_len):
            if temp == 0:
                if matches[i].item():
                    n_accepted += 1
                else:
                    break
            else:
                if uniforms[i].item() < accept_ratios[i].item():
                    n_accepted += 1
                else:
                    break

        print(f"[DFlash] n_accepted={n_accepted}/{verify_len}", flush=True)

        # 5. Rollback
        rollback = verify_len - n_accepted
        if rollback > 0:
            for c in batch.cache:
                if hasattr(c, 'offset'):
                    c.offset -= rollback
                elif hasattr(c, 'rollback'):
                    c.rollback(n_accepted)

        for i, c in enumerate(batch.cache):
            if hasattr(c, 'base'):
                batch.cache[i] = c.base

        # 6. Bonus/correction
        no_lp = mx.array(0.0)  # placeholder — no logprobs computed
        if n_accepted == verify_len:
            if temp == 0:
                bonus_val = all_next[verify_len].item()
            else:
                bonus_val = bonus_token.item()
        else:
            if temp == 0:
                bonus_val = all_next[n_accepted].item()
            else:
                bonus_val = corrections[n_accepted].item()

        # 7. Update state
        self._last_target_hidden[uid] = target_hidden[:, :n_accepted + 1, :]
        self._draft_position[uid] = start + n_accepted + 1

        # 8. Build token list (no logprobs)
        all_tokens = [(y_val, y_logprobs)]
        for i in range(n_accepted):
            all_tokens.append((drafts[i], no_lp))

        batch.y = mx.array([bonus_val])
        batch.logprobs = [no_lp]

        if n_accepted > 0:
            batch.tokens[0] = mx.concatenate(
                (batch.tokens[0], mx.array([t for t, _ in all_tokens[1:]])))
        batch.num_tokens[0] += len(all_tokens)

        # 9. Stop conditions
        toc = time.perf_counter()
        self._stats.generation_time += toc - tic
        self._stats.generation_tokens += len(all_tokens)

        stop_idx = None
        for idx, (tok, _) in enumerate(all_tokens):
            if tok in self.stop_tokens:
                stop_idx = idx
                break
            if batch.num_tokens[0] >= batch.max_tokens[0]:
                stop_idx = idx
                break

        first_tok, first_lp = all_tokens[0]

        if stop_idx is not None:
            valid_tokens = all_tokens[:stop_idx]
            if valid_tokens:
                if len(valid_tokens) > 1:
                    self._token_buffer[uid] = valid_tokens[1:]
                stop_tok, stop_lp = all_tokens[stop_idx]
                if uid not in self._token_buffer:
                    self._token_buffer[uid] = []
                self._token_buffer[uid].append((stop_tok, stop_lp))
                mx.async_eval(batch.y)
                return [self.Response(uid, first_tok, first_lp, None, lambda: None)]
            else:
                cache = batch.extract_cache(0)
                self.active_batch = None
                self._cleanup_uid(uid)
                return [self.Response(uid, first_tok, first_lp, "stop", cache)]

        if len(all_tokens) > 1:
            self._token_buffer[uid] = all_tokens[1:]

        mx.async_eval(batch.y)
        return [self.Response(uid, first_tok, first_lp, None, lambda: None)]

    def _yield_buffered(self, batch, uid):
        tic = time.perf_counter()
        buf = self._token_buffer[uid]
        tok, lp = buf.pop(0)
        if not buf:
            del self._token_buffer[uid]

        finish_reason = None
        if tok in self.stop_tokens:
            finish_reason = "stop"
        elif batch.num_tokens[0] >= batch.max_tokens[0]:
            finish_reason = "length"

        cache = None
        if finish_reason:
            cache = batch.extract_cache(0)
            self.active_batch = None
            self._cleanup_uid(uid)

        toc = time.perf_counter()
        self._stats.generation_time += toc - tic
        return [self.Response(uid, tok, lp, finish_reason, cache or (lambda: None))]

    def _cleanup_uid(self, uid):
        self._last_target_hidden.pop(uid, None)
        self._draft_position.pop(uid, None)
        self._prefilled.discard(uid)
        self._token_buffer.pop(uid, None)
        self._request_temp.pop(uid, None)
