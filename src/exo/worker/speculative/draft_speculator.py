from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from exo.shared.types.speculative import (
    DEFAULT_HORIZON_K,
    DEFAULT_TEMPERATURE,
    SpeculativeDraftCandidate,
    SpeculativeVerificationResult,
)


class EdgeDraftSpeculator:
    """Lightweight draft speculator driving fast local draft prediction on edge/weak nodes."""

    def __init__(self, sequence_id: str = "seq_default") -> None:
        self._sequence_id = sequence_id
        self._context_tokens: list[int] = []

    @property
    def current_context(self) -> Sequence[int]:
        return tuple(self._context_tokens)

    def set_initial_context(self, prompt_tokens: Sequence[int]) -> None:
        """Sets the initial prompt tokens."""
        self._context_tokens = list(prompt_tokens)

    def generate_draft_candidates(
        self,
        predict_next_logits_fn: Callable[[Sequence[int]], NDArray[np.float32]],
        horizon_k: int = DEFAULT_HORIZON_K,
        temperature: float = DEFAULT_TEMPERATURE,
        step: int = 0,
        random_generator: np.random.Generator | None = None,
    ) -> SpeculativeDraftCandidate:
        """Generates K candidate draft tokens using the local lightweight draft model."""
        rng = random_generator or np.random.default_rng()
        draft_tokens: list[int] = []
        draft_logprobs: list[float] = []

        working_context = list(self._context_tokens)

        for _ in range(horizon_k):
            logits = predict_next_logits_fn(working_context)

            # Softmax calculation
            shifted = logits - np.max(logits)
            if temperature <= 1e-5:
                sampled_token = int(np.argmax(logits))
                logprob = 0.0
            else:
                exp_vals = np.exp(shifted / temperature)
                probs = exp_vals / np.sum(exp_vals)
                sampled_token = int(rng.choice(len(probs), p=probs))
                logprob = float(np.log(max(probs[sampled_token], 1e-12)))

            draft_tokens.append(sampled_token)
            draft_logprobs.append(logprob)
            working_context.append(sampled_token)

        return SpeculativeDraftCandidate(
            sequence_id=self._sequence_id,
            step=step,
            draft_tokens=tuple(draft_tokens),
            draft_token_logprobs=tuple(draft_logprobs),
        )

    def commit_verification_result(self, result: SpeculativeVerificationResult) -> Sequence[int]:
        """Updates internal context with verified tokens, rolling back rejected hypotheses."""
        newly_emitted = result.all_emitted_tokens
        self._context_tokens.extend(newly_emitted)
        return newly_emitted
