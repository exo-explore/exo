from exo.shared.types.speculative import (
    SpeculativeConfig,
    SpeculativeVerificationResult,
)

EMA_DECAY = 0.8
HIGH_ACCEPTANCE_THRESHOLD = 0.80
LOW_ACCEPTANCE_THRESHOLD = 0.45


class AdaptiveHorizonScheduler:
    """Dynamically adjusts speculation horizon K based on rolling acceptance rates."""

    def __init__(self, config: SpeculativeConfig | None = None) -> None:
        self._config = config or SpeculativeConfig()
        self._current_k = self._config.default_horizon_k
        self._rolling_acceptance_rate: float = self._config.target_acceptance_rate
        self._total_draft_tokens: int = 0
        self._total_accepted_tokens: int = 0

    @property
    def current_horizon_k(self) -> int:
        return self._current_k

    @property
    def rolling_acceptance_rate(self) -> float:
        return self._rolling_acceptance_rate

    @property
    def cumulative_acceptance_rate(self) -> float:
        if self._total_draft_tokens == 0:
            return 0.0
        return self._total_accepted_tokens / self._total_draft_tokens

    @property
    def estimated_speedup_factor(self) -> float:
        """Estimates effective throughput speedup over pure autoregressive generation."""
        # Average accepted tokens per step + 1 correction token
        avg_tokens_per_step = (self.cumulative_acceptance_rate * self._current_k) + 1.0
        # Target verifier only runs 1 forward pass
        return max(1.0, avg_tokens_per_step)

    def record_verification(self, result: SpeculativeVerificationResult) -> int:
        """Updates rolling statistics and computes optimal horizon K for the next step."""
        num_draft = result.num_draft_tokens
        num_accepted = len(result.accepted_tokens)

        if num_draft > 0:
            current_rate = num_accepted / num_draft
            self._total_draft_tokens += num_draft
            self._total_accepted_tokens += num_accepted

            # Exponential moving average update
            self._rolling_acceptance_rate = (
                EMA_DECAY * self._rolling_acceptance_rate + (1.0 - EMA_DECAY) * current_rate
            )

            # Adapt horizon K
            if self._rolling_acceptance_rate > HIGH_ACCEPTANCE_THRESHOLD:
                self._current_k = min(self._config.max_horizon_k, self._current_k + 1)
            elif self._rolling_acceptance_rate < LOW_ACCEPTANCE_THRESHOLD:
                self._current_k = max(self._config.min_horizon_k, self._current_k - 1)

        return self._current_k
