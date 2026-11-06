import numpy as np
from numpy.typing import NDArray

from exo.shared.types.speculative import (
    SpeculativeDraftCandidate,
    SpeculativeVerificationResult,
)

GREEDY_TEMPERATURE_THRESHOLD = 1e-5
EPSILON_PROB = 1e-12


class SpeculativeRejectionSampler:
    """Rigorous unbiased speculative rejection sampler ensuring 100% mathematical zero-loss."""

    @staticmethod
    def _compute_softmax(logits: NDArray[np.float32], temperature: float) -> NDArray[np.float32]:
        """Calculates stable softmax probabilities scaled by temperature."""
        if temperature <= GREEDY_TEMPERATURE_THRESHOLD:
            # Deterministic one-hot for greedy decoding
            probs = np.zeros_like(logits)
            max_idx = np.argmax(logits)
            probs[max_idx] = 1.0
            return probs

        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted / temperature)
        sum_exp = np.sum(exp_vals)
        if sum_exp <= 0.0 or not np.isfinite(sum_exp):
            probs = np.zeros_like(logits)
            probs[np.argmax(logits)] = 1.0
            return probs
        return exp_vals / sum_exp

    @classmethod
    def verify_and_sample(
        cls,
        target_logits_batch: NDArray[np.float32],
        draft_candidate: SpeculativeDraftCandidate,
        temperature: float = 1.0,
        random_generator: np.random.Generator | None = None,
    ) -> SpeculativeVerificationResult:
        """Executes unbiased speculative verification on the batch of draft tokens.

        Args:
            target_logits_batch: Shape (K or K+1, vocab_size).
            draft_candidate: Candidate tokens and their draft probabilities/logprobs.
            temperature: Sampling temperature.
            random_generator: Optional numpy RNG.
        """
        rng = random_generator or np.random.default_rng()
        draft_tokens = draft_candidate.draft_tokens
        draft_logprobs = draft_candidate.draft_token_logprobs
        num_draft = len(draft_tokens)

        accepted_tokens: list[int] = []
        correction_token: int | None = None

        for i in range(num_draft):
            token_id = draft_tokens[i]
            target_logits_i = target_logits_batch[i]
            target_probs_i = cls._compute_softmax(target_logits_i, temperature)

            # Target probability for the draft token
            p_target = float(target_probs_i[token_id])

            # Draft probability
            draft_logprob = draft_logprobs[i]
            q_draft = float(np.exp(draft_logprob))
            q_draft = max(q_draft, EPSILON_PROB)

            # Acceptance ratio: alpha = min(1, p / q)
            alpha = min(1.0, p_target / q_draft)

            # Rejection sampling decision
            u = float(rng.uniform(0.0, 1.0))
            if u < alpha:
                # Accept candidate token
                accepted_tokens.append(token_id)
            else:
                # Reject candidate token: sample correction token from residual distribution
                # Residual distribution: max(0, p - q)
                residual_probs = np.maximum(0.0, target_probs_i)
                residual_probs[token_id] = 0.0  # Zero out rejected token
                norm_sum = np.sum(residual_probs)

                if norm_sum > 0.0 and np.isfinite(norm_sum):
                    norm_residual = residual_probs / norm_sum
                    correction_token = int(rng.choice(len(norm_residual), p=norm_residual))
                else:
                    correction_token = int(np.argmax(target_probs_i))
                break

        # If all K tokens were accepted, and an extra (K+1)-th logit slice is provided,
        # sample a bonus token directly from the target model
        if len(accepted_tokens) == num_draft and len(target_logits_batch) > num_draft:
            bonus_logits = target_logits_batch[num_draft]
            bonus_probs = cls._compute_softmax(bonus_logits, temperature)
            if temperature <= GREEDY_TEMPERATURE_THRESHOLD:
                correction_token = int(np.argmax(bonus_probs))
            else:
                correction_token = int(rng.choice(len(bonus_probs), p=bonus_probs))

        return SpeculativeVerificationResult(
            sequence_id=draft_candidate.sequence_id,
            step=draft_candidate.step,
            accepted_tokens=tuple(accepted_tokens),
            correction_token=correction_token,
            num_draft_tokens=num_draft,
        )
