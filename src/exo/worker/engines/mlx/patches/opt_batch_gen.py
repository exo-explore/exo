from typing import cast

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, GenerationBatch, PromptProcessingBatch

_PRECOMPUTE_TOP_K = 20

_original_step = GenerationBatch._step

_pending_topk_idx: mx.array | None = None
_pending_topk_val: mx.array | None = None
_pending_selected_lps: mx.array | None = None


def _patched_step(self: GenerationBatch) -> tuple[list[int], list[mx.array]]:
    self._current_tokens = self._next_tokens
    self._current_logprobs = self._next_logprobs
    inputs = self._current_tokens

    for i, ti in enumerate(self._token_context):
        self._token_context[i] = mx.concatenate(
            [ti[1:] if len(ti) == 256 else ti, inputs[i : i + 1]]
        )

    logits = self.model(inputs[:, None], cache=self.prompt_cache)
    logits = logits[:, -1, :]

    if self.logits_processors is not None and any(self.logits_processors):
        processed_logits: list[mx.array] = []
        for e in range(len(self.uids)):
            sample_logits = logits[e : e + 1]
            for processor in self.logits_processors[e]:
                sample_logits = processor(mx.array(self.tokens[e]), sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    if self.samplers is not None and any(self.samplers):
        all_samples: list[mx.array] = []
        for e in range(len(self.uids)):
            sample_sampler = self.samplers[e] or self.fallback_sampler
            all_samples.append(sample_sampler(logprobs[e : e + 1]))
        sampled = mx.concatenate(all_samples, axis=0)
    else:
        sampled = self.fallback_sampler(logprobs)

    self._next_tokens = sampled
    self._next_logprobs = list(logprobs)

    global _pending_topk_idx, _pending_topk_val, _pending_selected_lps

    needs_topk: bool = getattr(self, "_needs_topk", False)
    if needs_topk:
        batch_size = len(self.uids)
        k = min(_PRECOMPUTE_TOP_K, logprobs.shape[1])
        _pending_topk_idx = mx.argpartition(-logprobs, k, axis=1)[:, :k]
        _pending_topk_val = mx.take_along_axis(logprobs, _pending_topk_idx, axis=1)
        sort_order = mx.argsort(-_pending_topk_val, axis=1)
        _pending_topk_idx = mx.take_along_axis(_pending_topk_idx, sort_order, axis=1)
        _pending_topk_val = mx.take_along_axis(_pending_topk_val, sort_order, axis=1)
        _pending_selected_lps = logprobs[mx.arange(batch_size), sampled]
        mx.async_eval(
            self._next_tokens,
            *self._next_logprobs,
            *self._token_context,
            _pending_topk_idx,
            _pending_topk_val,
            _pending_selected_lps,
        )
    else:
        _pending_topk_idx = None
        _pending_topk_val = None
        _pending_selected_lps = None
        mx.async_eval(self._next_tokens, *self._next_logprobs, *self._token_context)

    mx.eval(inputs, *self._current_logprobs)
    token_list = cast(list[int], inputs.tolist())
    for sti, ti in zip(self.tokens, token_list, strict=True):
        sti.append(ti)
    return token_list, self._current_logprobs


_original_batch_next = BatchGenerator._next


def _fast_batch_next(
    self: BatchGenerator,
) -> tuple[list[PromptProcessingBatch.Response], list[GenerationBatch.Response]]:
    if (
        len(self._generation_batch) > 0
        and not self._unprocessed_sequences
        and len(self._prompt_batch) == 0
        and not self._currently_processing
    ):
        responses = self._generation_batch.next()
        self._gen_tokens_counter += len(responses)
        self._steps_counter += 1
        if self._steps_counter % 512 == 0:
            mx.clear_cache()
        return [], responses
    return _original_batch_next(self)


def apply_batch_gen_patch() -> None:
    GenerationBatch._step = _patched_step
    BatchGenerator._next = _fast_batch_next
