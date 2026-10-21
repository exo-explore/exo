from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from exo.shared.types.peripheral import LogitsPayload, SamplingConfig

# Minimum temperature threshold below which sampling falls back to argmax (greedy)
GREEDY_TEMPERATURE_THRESHOLD = 1e-5
NEGATIVE_INFINITY_LOGIT = -float("inf")


class PerTokenLogitsSampler:
    """High-efficiency, CPU-friendly vector sampler running on peripheral/edge nodes."""

    @staticmethod
    def apply_repetition_penalty(
        logits: NDArray[np.float32],
        generated_tokens: Sequence[int],
        penalty: float,
    ) -> NDArray[np.float32]:
        """Applies repetition penalty to logits of previously generated tokens."""
        if penalty == 1.0 or not generated_tokens:
            return logits

        result = logits.copy()
        # Set of unique tokens to penalize
        unique_tokens = set(generated_tokens)
        for token_id in unique_tokens:
            if 0 <= token_id < len(result):
                current_value = result[token_id]
                if current_value > 0:
                    result[token_id] = current_value / penalty
                else:
                    result[token_id] = current_value * penalty
        return result

    @staticmethod
    def apply_temperature(
        logits: NDArray[np.float32],
        temperature: float,
    ) -> NDArray[np.float32]:
        """Scales logits by temperature."""
        if temperature <= GREEDY_TEMPERATURE_THRESHOLD:
            return logits
        return logits / float(temperature)

    @staticmethod
    def apply_top_k(
        logits: NDArray[np.float32],
        top_k: int,
    ) -> NDArray[np.float32]:
        """Masks out all tokens below the top-k highest logits."""
        if top_k <= 0 or top_k >= len(logits):
            return logits

        result = logits.copy()
        # Find threshold value for top-k using partition
        cutoff_index = len(result) - top_k
        partitioned = np.partition(result, cutoff_index)
        threshold = partitioned[cutoff_index]
        result[result < threshold] = NEGATIVE_INFINITY_LOGIT
        return result

    @staticmethod
    def apply_top_p(
        logits: NDArray[np.float32],
        top_p: float,
    ) -> NDArray[np.float32]:
        """Nucleus sampling: masks tokens whose cumulative probability exceeds top-p."""
        if top_p >= 1.0 or top_p <= 0.0:
            return logits

        result = logits.copy()
        sorted_indices = np.argsort(result)[::-1]
        sorted_logits = result[sorted_indices]

        # Shift by max for numerical stability in softmax
        shifted = sorted_logits - np.max(sorted_logits)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)

        cumulative_probabilities = np.cumsum(probabilities)
        # Identify indices where cumulative probability exceeds top_p (preserving at least 1 token)
        mask = cumulative_probabilities > top_p
        mask[0] = False  # Keep at least the top candidate

        indices_to_remove = sorted_indices[mask]
        result[indices_to_remove] = NEGATIVE_INFINITY_LOGIT
        return result

    @staticmethod
    def apply_min_p(
        logits: NDArray[np.float32],
        min_p: float,
    ) -> NDArray[np.float32]:
        """Min-p sampling: masks out tokens with probability < max_prob * min_p."""
        if min_p <= 0.0:
            return logits

        result = logits.copy()
        shifted = result - np.max(result)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)

        max_prob = np.max(probabilities)
        threshold = max_prob * min_p
        result[probabilities < threshold] = NEGATIVE_INFINITY_LOGIT
        return result

    @classmethod
    def sample_from_dense_logits(
        cls,
        raw_logits: NDArray[np.float32],
        config: SamplingConfig,
        generated_tokens: Sequence[int] | None = None,
        random_generator: np.random.Generator | None = None,
    ) -> tuple[int, float]:
        """Performs end-to-end token sampling from raw dense logits.

        Returns:
            tuple of (sampled_token_id, token_log_probability)
        """
        rng = random_generator or (
            np.random.default_rng(config.seed)
            if config.seed is not None
            else np.random.default_rng()
        )

        working_logits = raw_logits.astype(np.float32, copy=True)

        # 1. Repetition penalty
        if generated_tokens and config.repetition_penalty != 1.0:
            working_logits = cls.apply_repetition_penalty(
                working_logits, generated_tokens, config.repetition_penalty
            )

        # 2. Greedy decoding check
        if config.temperature <= GREEDY_TEMPERATURE_THRESHOLD:
            sampled_id = int(np.argmax(working_logits))
            shifted = working_logits - np.max(working_logits)
            logprob = float(shifted[sampled_id] - np.log(np.sum(np.exp(shifted))))
            return sampled_id, logprob

        # 3. Temperature scaling
        working_logits = cls.apply_temperature(working_logits, config.temperature)

        # 4. Top-K filtering
        if config.top_k > 0:
            working_logits = cls.apply_top_k(working_logits, config.top_k)

        # 5. Top-P (Nucleus) filtering
        if config.top_p < 1.0:
            working_logits = cls.apply_top_p(working_logits, config.top_p)

        # 6. Min-P filtering
        if config.min_p > 0.0:
            working_logits = cls.apply_min_p(working_logits, config.min_p)

        # 7. Probability distribution and categorical draw
        finite_mask = np.isfinite(working_logits)
        if not np.any(finite_mask):
            # Fallback to pure argmax of original logits if all filtered
            fallback_id = int(np.argmax(raw_logits))
            return fallback_id, 0.0

        shifted_finite = working_logits[finite_mask] - np.max(working_logits[finite_mask])
        exp_finite = np.exp(shifted_finite)
        probs_finite = exp_finite / np.sum(exp_finite)

        finite_indices = np.where(finite_mask)[0]
        sampled_id = int(rng.choice(finite_indices, p=probs_finite))
        sampled_logprob = float(np.log(probs_finite[finite_indices == sampled_id][0]))

        return sampled_id, sampled_logprob

    @classmethod
    def reconstruct_dense_logits(cls, payload: LogitsPayload) -> NDArray[np.float32]:
        """Reconstructs a dense logits array from a potentially sparse logits payload."""
        dense = np.full(payload.vocabulary_size, NEGATIVE_INFINITY_LOGIT, dtype=np.float32)
        for entry in payload.sparse_entries:
            if 0 <= entry.token_index < payload.vocabulary_size:
                dense[entry.token_index] = entry.logit_value
        return dense
