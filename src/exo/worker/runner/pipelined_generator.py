"""Multi-stream pipelined batch generator for pipeline-parallel inference.

When a model is split across N ranks (pipeline parallelism), each rank's GPU is idle
for (N-1)/N of each step while waiting for other ranks to compute their layers.

This module fills the pipeline bubble by splitting sequences into N micro-batch groups
and processing each group on a different MLX stream. The GPU can overlap one stream's
network communication (send/recv/all_gather) with another stream's compute.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAny=false

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache


@dataclass
class MicroBatch:
    """State for one micro-batch group of sequences."""

    uids: list[int]
    y: mx.array  # Last sampled tokens [batch]
    logprobs: list[mx.array]  # Logprobs for each sequence
    max_tokens: list[int]
    num_tokens: list[int]
    cache: list[Any]  # KV cache (list of layer caches)
    samplers: list[Callable[[mx.array], mx.array]]
    tokens: list[mx.array]  # All tokens generated so far per sequence

    def __len__(self) -> int:
        return len(self.uids)


@dataclass
class PipelinedResponse:
    """Response from one generation step."""

    uid: int
    token: int
    logprobs: mx.array
    finish_reason: str | None
    cache: list[Any] | None = None


@dataclass
class PendingPrompt:
    """A prompt waiting to be prefilled."""

    uid: int
    tokens: list[int]
    max_tokens: int
    sampler: Callable[[mx.array], mx.array]


class PipelinedGenerator:
    """
    Multi-stream batch generator that fills pipeline bubbles.

    Splits active sequences into `world_size` micro-batch groups, each processed
    on its own MLX stream. During mx.eval(), the GPU overlaps network operations
    on one stream with compute on another.
    """

    def __init__(
        self,
        model: nn.Module,
        world_size: int,
        stop_tokens: set[int] | None = None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.world_size = world_size
        self.stop_tokens = stop_tokens or set()
        self.max_tokens_default = max_tokens

        # Create one stream per pipeline stage
        self.streams = [mx.new_stream(mx.default_device()) for _ in range(world_size)]

        # Micro-batch groups (one per stream)
        self.micro_batches: list[MicroBatch | None] = [None] * world_size

        # Pending prompts to be inserted
        self.pending_prompts: list[PendingPrompt] = []

        # UID counter
        self._next_uid = 0

    @property
    def active_count(self) -> int:
        """Total number of active sequences across all micro-batches."""
        return sum(len(mb) for mb in self.micro_batches if mb is not None)

    @property
    def has_active(self) -> bool:
        return self.active_count > 0

    def insert(
        self,
        prompts: list[list[int]],
        max_tokens: list[int],
        samplers: list[Callable[[mx.array], mx.array]],
    ) -> list[int]:
        """Queue prompts for processing. Returns assigned UIDs."""
        uids: list[int] = []
        for prompt, mt, sampler in zip(prompts, max_tokens, samplers, strict=True):
            uid = self._next_uid
            self._next_uid += 1
            self.pending_prompts.append(
                PendingPrompt(uid=uid, tokens=prompt, max_tokens=mt, sampler=sampler)
            )
            uids.append(uid)
        return uids

    def _prefill_group(self, group_idx: int, prompts: list[PendingPrompt]) -> None:
        """Prefill a group of prompts and create a MicroBatch."""
        if not prompts:
            return

        stream = self.streams[group_idx]

        with mx.stream(stream):
            # Create per-sequence caches
            caches = [make_prompt_cache(self.model) for _ in prompts]

            # Tokenize and prefill each sequence
            all_y: list[mx.array] = []
            all_logprobs: list[mx.array] = []
            all_samplers: list[Callable[[mx.array], mx.array]] = []
            all_tokens: list[mx.array] = []

            for prompt_info, cache in zip(prompts, caches, strict=True):
                tokens = mx.array(prompt_info.tokens)
                # Run prefill (process all tokens except last)
                if len(prompt_info.tokens) > 1:
                    self.model(tokens[:-1][None, :], cache=cache)
                    mx.eval([c.state for c in cache])

                # Process last token to get first generation logits
                last_token = tokens[-1:][None, :]
                logits = self.model(last_token, cache=cache)
                logits = logits[:, -1, :]
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                sampled = prompt_info.sampler(logprobs)

                all_y.append(sampled.squeeze(0))
                all_logprobs.append(logprobs.squeeze(0))
                all_samplers.append(prompt_info.sampler)
                all_tokens.append(tokens)

            mx.eval(*all_y, *all_logprobs)

        # Create micro-batch
        batch = MicroBatch(
            uids=[p.uid for p in prompts],
            y=mx.stack(all_y),
            logprobs=all_logprobs,
            max_tokens=[p.max_tokens for p in prompts],
            num_tokens=[0] * len(prompts),
            cache=caches,
            samplers=all_samplers,
            tokens=all_tokens,
        )

        if self.micro_batches[group_idx] is None:
            self.micro_batches[group_idx] = batch
        else:
            # Extend existing micro-batch (would need cache merging - for now replace)
            self.micro_batches[group_idx] = batch

    def _prefill_pending(self) -> None:
        """Distribute pending prompts across micro-batch groups and prefill."""
        if not self.pending_prompts:
            return

        # Distribute round-robin across groups
        groups: list[list[PendingPrompt]] = [[] for _ in range(self.world_size)]
        for i, prompt in enumerate(self.pending_prompts):
            groups[i % self.world_size].append(prompt)
        self.pending_prompts.clear()

        for group_idx, group_prompts in enumerate(groups):
            if group_prompts:
                self._prefill_group(group_idx, group_prompts)

    def _step_all(self) -> None:
        """
        Run one generation step across all micro-batch groups on different streams.

        This is where pipeline overlap happens: each group's model forward pass
        runs on its own stream, and mx.eval() allows the GPU to overlap network
        ops (send/recv/all_gather) from one stream with compute from another.
        """
        # Build computation graphs on each stream (lazy, no evaluation yet)
        new_y_list: list[mx.array] = []
        new_logprobs_list: list[list[mx.array]] = []
        active_indices: list[int] = []

        for i, mb in enumerate(self.micro_batches):
            if mb is None or len(mb) == 0:
                continue
            active_indices.append(i)

            with mx.stream(self.streams[i]):
                # Prepare input: last sampled tokens
                input_tokens = mb.y[:, None]  # [batch, 1]

                # Forward pass (lazy graph construction)
                # For pipeline models, this includes send/recv/all_gather ops
                logits = self.model(input_tokens, cache=mb.cache)
                logits = logits[:, -1, :]  # [batch, vocab]

                # Compute logprobs
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

                # Sample per-sequence
                batch_size = len(mb)
                if batch_size == 1:
                    sampled = mb.samplers[0](logprobs)
                else:
                    samples = []
                    for e in range(batch_size):
                        samples.append(mb.samplers[e](logprobs[e: e + 1]))
                    sampled = mx.concatenate(samples, axis=0)

                new_y_list.append(sampled)
                new_logprobs_list.append([logprobs[e] for e in range(batch_size)])

        if not active_indices:
            return

        # Evaluate ALL streams together - this is where overlap happens!
        # The GPU can execute stream0's all_gather while computing stream1's layers.
        mx.eval(*new_y_list, *[lp for lps in new_logprobs_list for lp in lps])

        # Update micro-batch states with results
        for list_idx, group_idx in enumerate(active_indices):
            mb = self.micro_batches[group_idx]
            assert mb is not None
            mb.y = new_y_list[list_idx]
            mb.logprobs = new_logprobs_list[list_idx]
            # Append sampled tokens to history
            for e in range(len(mb)):
                mb.tokens[e] = mx.concatenate([mb.tokens[e], mb.y[e: e + 1]])

    def next(self) -> list[PipelinedResponse]:
        """
        Run one generation step and return responses.

        Returns a PipelinedResponse for each active sequence (across all groups).
        Finished sequences are removed from their micro-batch.
        """
        # Prefill any pending prompts first
        self._prefill_pending()

        if not self.has_active:
            return []

        # Run the multi-stream forward pass
        self._step_all()

        # Collect responses and filter completed sequences
        responses: list[PipelinedResponse] = []

        for group_idx, mb in enumerate(self.micro_batches):
            if mb is None or len(mb) == 0:
                continue

            keep_idx: list[int] = []
            end_idx: list[int] = []

            for e in range(len(mb)):
                token = int(mb.y[e].item())
                uid = mb.uids[e]
                num_tok = mb.num_tokens[e] + 1
                max_tok = mb.max_tokens[e]
                mb.num_tokens[e] = num_tok

                if token in self.stop_tokens:
                    finish_reason = "stop"
                    end_idx.append(e)
                elif num_tok >= max_tok:
                    finish_reason = "length"
                    end_idx.append(e)
                else:
                    finish_reason = None
                    keep_idx.append(e)

                responses.append(
                    PipelinedResponse(
                        uid=uid,
                        token=token,
                        logprobs=mb.logprobs[e],
                        finish_reason=finish_reason,
                    )
                )

            # Remove finished sequences
            if end_idx:
                if keep_idx:
                    # Filter the micro-batch to keep only active sequences
                    mb.uids = [mb.uids[i] for i in keep_idx]
                    mb.y = mb.y[mx.array(keep_idx)]
                    mb.logprobs = [mb.logprobs[i] for i in keep_idx]
                    mb.max_tokens = [mb.max_tokens[i] for i in keep_idx]
                    mb.num_tokens = [mb.num_tokens[i] for i in keep_idx]
                    mb.samplers = [mb.samplers[i] for i in keep_idx]
                    mb.tokens = [mb.tokens[i] for i in keep_idx]
                    # Cache filtering: trim batch dimension
                    for c in mb.cache:
                        if hasattr(c, "keys") and c.keys is not None:
                            c.keys = c.keys[mx.array(keep_idx)]
                            c.values = c.values[mx.array(keep_idx)]
                else:
                    self.micro_batches[group_idx] = None

        return responses

    def close(self) -> None:
        """Clean up resources."""
        self.micro_batches = [None] * self.world_size
        self.pending_prompts.clear()
