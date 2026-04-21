"""Rank-aware DFlashBatchGenerator patches for the attn/MoE split.

Both ranks draft independently, then sync on MOE_RANK's drafts before the
pipelined verify forward. Acceptance is deterministic at temp=0 (both ranks
get the same verify_logits via the pipelined forward, so both compute the
same n_accepted). At temp>0, MOE_RANK's n_accepted wins via all_gather so
stochastic decisions stay consistent.

Stock `_first_step_capture` is kept — it reads `self._captured['prefill_hiddens']`
which is populated by `_CapturingLayer.__call__` during prefill. Our
pipelined prefill bypasses `layer.__call__` entirely, so `_CapturingLayer`
would never fire on its own. `make_pipelined_model_call` compensates by
detecting the `_CapturingLayer` wrappers and writing the captured layer
outputs from `pipelined_layer_loop` directly into the closure-captured
`captured` dict (see `_populate_dflash_captured` in `model_forward.py`).
"""

import time

import mlx.core as mx

from .decoder import ATTN_RANK, MOE_RANK


def make_split_speculative_next(group):  # type: ignore[no-untyped-def]
    """Build a DFlashBatchGenerator._speculative_next replacement closed over group."""
    rank = group.rank()

    def _split_speculative_next(self):  # type: ignore[no-untyped-def]
        from exo.worker.engines.mlx.speculative.dflash_speculative import (
            dflash_speculative_forward,
        )

        tic = time.perf_counter()
        batch = self.active_batch
        uid = batch.uids[0]
        y = batch.y
        y_val = y[0].item()
        y_logprobs = batch.logprobs[0]

        batch.tokens[0] = mx.concatenate((batch.tokens[0], y[0:1]))

        last_target_hidden = self._last_target_hidden.get(uid)
        if last_target_hidden is None:
            print(
                f"[rank {rank}] DFlash: NO target_hidden -> fallback to super()._next() "
                f"(y_val={y_val})",
                flush=True,
            )
            return super(type(self), self)._next()
        print(
            f"[rank {rank}] DFlash speculative cycle "
            f"(y_val={y_val}, target_hidden.shape={last_target_hidden.shape})",
            flush=True,
        )

        bs = self.drafter.block_size
        verify_len = self.verify_len
        temp = self._request_temp.get(uid, self.temp)
        alpha = self.alpha

        # ATTN_RANK's cache offset is correct (updated by attention); MOE_RANK's
        # stays at 0 because it never runs attention. Sync _draft_position
        # across ranks so the drafter uses the right positional encoding.
        local_start = self._draft_position[uid]
        gathered = mx.distributed.all_gather(
            mx.array([local_start], dtype=mx.int32), group=group
        )
        start = int(gathered[ATTN_RANK].item())
        self._draft_position[uid] = start

        # 1. Draft — both ranks draft independently
        block_ids = mx.full((1, bs), self.drafter.mask_token_id, dtype=mx.int32)
        block_ids[:, 0] = y_val
        draft_logits = self.drafter.draft(last_target_hidden, block_ids, start)
        self.drafter.crop_draft_cache(start)

        # 2. Sample — both ranks sample locally
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

        # 3. Sync drafts: take MOE_RANK's drafts as source of truth for both ranks.
        drafts_local = mx.array([all_drafts[:verify_len]], dtype=mx.int32)  # (1, V)
        gathered = mx.distributed.all_gather(drafts_local, group=group)      # (2, V)
        drafts_arr = gathered[MOE_RANK : MOE_RANK + 1]                       # (1, V)
        mx.eval(drafts_arr)
        drafts = drafts_arr[0].tolist()

        # 4. Build verify_input on BOTH ranks (identical tokens now)
        y_val_tensor = mx.array([[y_val]], dtype=mx.int32)
        verify_input = mx.concatenate([y_val_tensor, drafts_arr], axis=1)

        # 5. Pipelined verify forward — both ranks run the 2N+1 stage pipeline
        #    via the patched dflash_speculative_forward.
        target_hidden, _, verify_logits = dflash_speculative_forward(
            self.model,
            verify_input,
            batch.cache,
            self.drafter.target_layer_ids,
            speculative=True,
        )

        # 6. Acceptance. temp==0: deterministic on both ranks.
        #    temp>0: MOE_RANK decides, all_gather broadcasts n_accepted.
        if temp == 0:
            target_tokens = mx.argmax(verify_logits[:, :verify_len, :], axis=-1)
            matches = mx.equal(target_tokens, drafts_arr).squeeze(0)
            all_next = mx.argmax(verify_logits[0], axis=-1)
            mx.eval(matches, all_next, target_hidden)
            n_accepted = 0
            for i in range(verify_len):
                if matches[i].item():
                    n_accepted += 1
                else:
                    break
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
            bonus_token = mx.random.categorical(
                verify_logits[0, verify_len] * (1.0 / temp)
            )
            mx.async_eval(
                accept_ratios, uniforms, corrections, bonus_token, target_hidden
            )
            # Only MOE_RANK decides; broadcast n_accepted.
            if rank == MOE_RANK:
                n_accepted_local = 0
                for i in range(verify_len):
                    if uniforms[i].item() < accept_ratios[i].item():
                        n_accepted_local += 1
                    else:
                        break
            else:
                n_accepted_local = 0
            gathered_n = mx.distributed.all_gather(
                mx.array([n_accepted_local], dtype=mx.int32), group=group
            )
            n_accepted = int(gathered_n[MOE_RANK].item())

        if rank == MOE_RANK:
            print(
                f"[DFlash] n_accepted={n_accepted}/{verify_len}",
                flush=True,
            )

        # 7. Rollback — same n_accepted on both ranks keeps caches consistent
        rollback = verify_len - n_accepted
        if rollback > 0:
            for c in batch.cache:
                if hasattr(c, "offset"):
                    c.offset -= rollback
                elif hasattr(c, "rollback"):
                    c.rollback(n_accepted)

        for i, c in enumerate(batch.cache):
            if hasattr(c, "base"):
                batch.cache[i] = c.base

        # 8. Bonus / correction token
        no_lp = mx.array(0.0)
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

        # 9. Update state
        self._last_target_hidden[uid] = target_hidden[:, : n_accepted + 1, :]
        self._draft_position[uid] = start + n_accepted + 1

        # 10. Build token list (no logprobs)
        all_tokens = [(y_val, y_logprobs)]
        for i in range(n_accepted):
            all_tokens.append((drafts[i], no_lp))

        batch.y = mx.array([bonus_val])
        batch.logprobs = [no_lp]

        if n_accepted > 0:
            batch.tokens[0] = mx.concatenate(
                (batch.tokens[0], mx.array([t for t, _ in all_tokens[1:]]))
            )
        batch.num_tokens[0] += len(all_tokens)

        # 11. Stop conditions (same as stock)
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

    return _split_speculative_next
