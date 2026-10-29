from collections.abc import Mapping, Sequence

from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel

DEFAULT_HORIZON_K = 4
MIN_HORIZON_K = 1
MAX_HORIZON_K = 8
DEFAULT_TEMPERATURE = 1.0


class SpeculativeConfig(FrozenModel):
    """Configuration for heterogeneous speculative decoding."""

    default_horizon_k: int = DEFAULT_HORIZON_K
    min_horizon_k: int = MIN_HORIZON_K
    max_horizon_k: int = MAX_HORIZON_K
    temperature: float = DEFAULT_TEMPERATURE
    target_acceptance_rate: float = 0.65


class SpeculativeDraftCandidate(FrozenModel):
    """Candidate draft sequence produced by the edge draft model."""

    sequence_id: str
    step: int
    draft_tokens: Sequence[int]
    draft_token_logprobs: Sequence[float]
    metadata: Mapping[str, str] = Field(default_factory=dict)

    @property
    def horizon_length(self) -> int:
        return len(self.draft_tokens)


class SpeculativeVerificationResult(FrozenModel):
    """Result emitted by the target verifier model after speculative rejection sampling."""

    sequence_id: str
    step: int
    accepted_tokens: Sequence[int]
    correction_token: int | None = None
    num_draft_tokens: int

    @property
    def acceptance_rate(self) -> float:
        if self.num_draft_tokens == 0:
            return 0.0
        return len(self.accepted_tokens) / self.num_draft_tokens

    @property
    def all_emitted_tokens(self) -> Sequence[int]:
        """Returns the full sequence of approved tokens for this step."""
        if self.correction_token is not None:
            return (*self.accepted_tokens, self.correction_token)
        return self.accepted_tokens
