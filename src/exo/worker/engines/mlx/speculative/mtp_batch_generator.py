#!/usr/bin/env python3
"""MTP Speculative Decoding integrated with mlx_lm's BatchGenerator.

Subclasses BatchGenerator to add MTP drafting + S>1 verification with
correct GDN state rollback via SpeculativeArraysCache.

At BS=1: drafts γ tokens with MTP, verifies at S=γ+1, buffers accepted tokens.
At BS>1: falls back to standard BatchGenerator (no speculative).

Usage:
    from mtp_batch_generator import MTPBatchGenerator
    gen = MTPBatchGenerator(model, mtp_predictor, gamma=2, ...)
    gen.insert([prompt_tokens])
    while True:
        responses = gen.next()
"""

import time

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, generation_stream

from .mtp_module import MTPPredictor, speculative_forward, draft_tokens


class MTPBatchGenerator(BatchGenerator):
    """BatchGenerator with MTP speculative decoding for BS=1."""

    def __init__(
        self,
        model,
        mtp_predictor: MTPPredictor,
        gamma: int = 2,
        temp: float = 0.0,
        alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.mtp = mtp_predictor
        self.gamma = gamma
        self.temp = temp
        self.alpha = alpha

        self._token_buffer = {}       # uid → [(token, logprobs), ...]
        self._captured = {}            # pre_norm / prompt_pre_norm from norm wrapper
        self._mtp_pre_norm = {}        # uid → (B, 1, D) pre-norm hidden state
        self._mtp_prefilled = set()    # uids with MTP cache prefilled
        self._request_temp = {}        # uid → temperature from request

        self._setup_hidden_capture()

    def _setup_hidden_capture(self):
        """Monkey-patch model's final norm to capture pre-norm hidden state.

        Captures:
        - pre_norm: hidden states before final RMSNorm (for MTP input)
        - prompt_pre_norm: same but only when S>1 (prefill)
        """
        inner = getattr(self.model, 'model', None) or self.model.language_model.model
        original_norm = inner.norm
        captured = self._captured

        class _CapturingNorm:
            def __init__(self, orig):
                self._orig = orig
                self.weight = orig.weight

            def __call__(self, x):
                captured['pre_norm'] = x
                if x.shape[1] > 1:
                    captured['prompt_pre_norm'] = x
                return self._orig(x)

            def __getattr__(self, name):
                return getattr(self._orig, name)

        inner.norm = _CapturingNorm(original_norm)

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
                and self.gamma > 0
                and len(self.unprocessed_prompts) == 0):
            uid = batch.uids[0]
            if uid not in self._mtp_prefilled:
                print(f"[MTP] first_step_and_prefill uid={uid}")
                return self._first_step_and_prefill(batch, uid)
            print(f"[MTP] speculative_next uid={uid} buf={len(self._token_buffer.get(uid, []))}")
            return self._speculative_next()

        # Standard path (BS>1 or no batch)
        print(f"[MTP] standard path batch={batch is not None} len={len(batch) if batch else 0} unprocessed={len(self.unprocessed_prompts)}")
        responses = super()._next()
        if responses and batch is not None and len(batch) == 1:
            if 'pre_norm' in self._captured:
                uid = batch.uids[0]
                self._mtp_pre_norm[uid] = self._captured['pre_norm'][:, -1:, :]
        return responses

    def _first_step_and_prefill(self, batch, uid):
        """First decode step. MTP cache already prefilled by ExoBatchGenerator.submit()."""
        responses = super()._next()
        if not responses:
            return responses

        # Capture decode pre_norm from this standard step for first speculative cycle
        decode_pre_norm = self._captured.get('pre_norm')
        if decode_pre_norm is not None:
            mx.eval(decode_pre_norm)
            self._mtp_pre_norm[uid] = decode_pre_norm[:, -1:, :]

        self._mtp_prefilled.add(uid)
        return responses

    def _speculative_next(self):
        """Core speculative cycle with correct GDN rollback."""
        tic = time.perf_counter()
        batch = self.active_batch
        uid = batch.uids[0]
        y = batch.y          # (1,) — token from previous step, to be yielded
        y_val = y[0].item()
        y_logprobs = batch.logprobs[0]

        # Append current y to token history
        batch.tokens[0] = mx.concatenate((batch.tokens[0], y[0:1]))

        pre_norm = self._mtp_pre_norm.get(uid)
        if pre_norm is None:
            return super()._next()

        gamma = self.gamma
        temp = self._request_temp.get(uid, self.temp)
        alpha = self.alpha

        # 1. Draft γ tokens (lazy chain, no eval)
        next_token_arr = y.reshape(1, 1)
        draft_ids, draft_probs = draft_tokens(
            self.mtp, pre_norm, next_token_arr, gamma, temp)

        # 2. Verify via speculative_forward (handles GDN cache wrapping + kernel swap)
        draft_concat = mx.concatenate(
            [d.reshape(1, 1) for d in draft_ids], axis=1)  # (1, γ)
        verify_input = mx.concatenate(
            [next_token_arr, draft_concat], axis=1)  # (1, γ+1)
        verify_pre_norm, verify_logits = speculative_forward(
            self.model, verify_input, batch.cache, speculative=True)

        # 3. Build acceptance check lazily
        target_tokens = mx.argmax(verify_logits[:, :gamma, :], axis=-1)

        if temp == 0:
            matches = mx.equal(target_tokens, draft_concat).squeeze(0)
            all_next = mx.argmax(verify_logits[0], axis=-1)
            logprobs_all = verify_logits[0] - mx.logsumexp(
                verify_logits[0], axis=-1, keepdims=True)
            mx.async_eval(matches, all_next, logprobs_all, verify_pre_norm)
        else:
            accept_ratios = []
            for i in range(gamma):
                p = mx.softmax(verify_logits[0, i] / temp, axis=-1)
                q = draft_probs[i]
                p_di = p[draft_ids[i].squeeze()]
                q_di = q[0, draft_ids[i].squeeze()]
                ratio = p_di / mx.maximum(q_di, 1e-10)
                accept_ratios.append(mx.minimum(ratio ** alpha, 1.0))
            uniforms = mx.random.uniform(shape=(gamma,))
            corrections = []
            for i in range(gamma):
                p = mx.softmax(verify_logits[0, i] / temp, axis=-1)
                q = draft_probs[i][0]
                residual = mx.maximum(p - q, 0.0)
                corrections.append(mx.random.categorical(mx.log(residual + 1e-10)))
            bonus_token = mx.random.categorical(verify_logits[0, gamma] * (1.0 / temp))
            logprobs_all = verify_logits[0] - mx.logsumexp(
                verify_logits[0], axis=-1, keepdims=True)
            mx.async_eval(accept_ratios, uniforms, corrections, bonus_token,
                          logprobs_all, verify_pre_norm, draft_concat)

        # 4. Determine acceptance
        n_accepted = 0
        for i in range(gamma):
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

        # 5. Rollback cache
        rollback = gamma - n_accepted
        if rollback > 0:
            for c in batch.cache:
                if hasattr(c, 'offset'):
                    c.offset -= rollback
                elif hasattr(c, 'rollback'):
                    c.rollback(n_accepted)

        # Unwrap SpeculativeArraysCache
        for i, c in enumerate(batch.cache):
            if hasattr(c, 'base'):
                batch.cache[i] = c.base

        # 6. Bonus/correction token + logprobs
        if n_accepted == gamma:
            if temp == 0:
                bonus_val = all_next[gamma].item()
            else:
                bonus_val = bonus_token.item()
            bonus_lp = logprobs_all[gamma]
        else:
            if temp == 0:
                bonus_val = all_next[n_accepted].item()
            else:
                bonus_val = corrections[n_accepted].item()
            bonus_lp = logprobs_all[n_accepted]

        # 7. Update MTP pre_norm for next cycle
        self._mtp_pre_norm[uid] = verify_pre_norm[
            :, (gamma if n_accepted == gamma else n_accepted):
               (gamma if n_accepted == gamma else n_accepted) + 1, :]

        # 8. Build token list: current y + accepted drafts
        draft_int_values = draft_concat[0].tolist()
        all_tokens = [(y_val, y_logprobs)]
        for i in range(n_accepted):
            all_tokens.append((draft_int_values[i], logprobs_all[i]))

        # 9. Set batch.y = bonus for next cycle
        batch.y = mx.array([bonus_val])
        batch.logprobs = [bonus_lp]

        # Append accepted drafts to token history
        if n_accepted > 0:
            batch.tokens[0] = mx.concatenate(
                (batch.tokens[0], mx.array([t for t, _ in all_tokens[1:]])))
        batch.num_tokens[0] += len(all_tokens)

        # 10. Check stop conditions — truncate at stop token
        toc = time.perf_counter()
        self._stats.generation_time += toc - tic
        self._stats.generation_tokens += len(all_tokens)

        # Find first stop token or length limit in all_tokens
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
            # Tokens before the stop are valid output — buffer them
            # The stop token itself triggers finish_reason
            valid_tokens = all_tokens[:stop_idx]
            if valid_tokens:
                # Yield first, buffer rest + a final stop entry
                if len(valid_tokens) > 1:
                    self._token_buffer[uid] = valid_tokens[1:]
                # Append stop marker as last buffered token
                stop_tok, stop_lp = all_tokens[stop_idx]
                if uid not in self._token_buffer:
                    self._token_buffer[uid] = []
                self._token_buffer[uid].append((stop_tok, stop_lp))
                mx.async_eval(batch.y)
                return [self.Response(uid, first_tok, first_lp, None, lambda: None)]
            else:
                # Stop token is the first token — finish immediately
                cache = batch.extract_cache(0)
                self.active_batch = None
                self._cleanup_uid(uid)
                return [self.Response(uid, first_tok, first_lp, "stop", cache)]

        # Buffer remaining tokens
        if len(all_tokens) > 1:
            self._token_buffer[uid] = all_tokens[1:]

        mx.async_eval(batch.y)
        return [self.Response(uid, first_tok, first_lp, None, lambda: None)]

    def _yield_buffered(self, batch, uid):
        """Yield one buffered token from a previous speculative cycle."""
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
        """Clean up MTP state for a finished request."""
        self._mtp_pre_norm.pop(uid, None)
        self._mtp_prefilled.discard(uid)
        self._token_buffer.pop(uid, None)
        self._request_temp.pop(uid, None)
