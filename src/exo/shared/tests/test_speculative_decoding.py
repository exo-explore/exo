import numpy as np
from numpy.typing import NDArray

from exo.shared.types.speculative import (
    SpeculativeConfig,
    SpeculativeDraftCandidate,
    SpeculativeVerificationResult,
)
from exo.worker.speculative.adaptive_horizon import AdaptiveHorizonScheduler
from exo.worker.speculative.draft_speculator import EdgeDraftSpeculator
from exo.worker.speculative.rejection_sampler import SpeculativeRejectionSampler


def test_speculative_rejection_sampler_perfect_agreement() -> None:
    # 3 draft tokens: [10, 20, 30]
    draft_tokens = (10, 20, 30)
    # Draft model assigns high probability to these tokens
    draft_logprobs = (float(np.log(0.9)), float(np.log(0.85)), float(np.log(0.95)))

    candidate = SpeculativeDraftCandidate(
        sequence_id="seq-test",
        step=1,
        draft_tokens=draft_tokens,
        draft_token_logprobs=draft_logprobs,
    )

    # Target model also assigns dominant probability to the same tokens
    vocab_size = 50
    target_logits = np.zeros((4, vocab_size), dtype=np.float32)
    target_logits[0, 10] = 10.0
    target_logits[1, 20] = 10.0
    target_logits[2, 30] = 10.0
    target_logits[3, 40] = 10.0  # Bonus token slice

    result = SpeculativeRejectionSampler.verify_and_sample(
        target_logits_batch=target_logits,
        draft_candidate=candidate,
        temperature=1.0,
    )

    # All 3 candidate tokens must be accepted
    assert result.accepted_tokens == (10, 20, 30)
    assert result.acceptance_rate == 1.0
    # Bonus token sampled from the 4th slice
    assert result.correction_token == 40
    assert result.all_emitted_tokens == (10, 20, 30, 40)


def test_speculative_rejection_sampler_immediate_rejection() -> None:
    # Draft model guesses token 5, but Target model strongly predicts token 9
    draft_tokens = (5, 6, 7)
    draft_logprobs = (float(np.log(0.9)), float(np.log(0.9)), float(np.log(0.9)))

    candidate = SpeculativeDraftCandidate(
        sequence_id="seq-test",
        step=1,
        draft_tokens=draft_tokens,
        draft_token_logprobs=draft_logprobs,
    )

    vocab_size = 20
    target_logits = np.zeros((3, vocab_size), dtype=np.float32)
    # At position 0, token 5 has logit -10 (probability ~0), while token 9 has logit 10 (probability ~1)
    target_logits[0, 5] = -10.0
    target_logits[0, 9] = 10.0

    result = SpeculativeRejectionSampler.verify_and_sample(
        target_logits_batch=target_logits,
        draft_candidate=candidate,
        temperature=1.0,
    )

    # First token rejected -> 0 tokens accepted
    assert len(result.accepted_tokens) == 0
    assert result.acceptance_rate == 0.0
    # Correction token should be sampled from target distribution (token 9)
    assert result.correction_token == 9
    assert result.all_emitted_tokens == (9,)


def test_speculative_rejection_sampler_greedy_mode() -> None:
    draft_tokens = (1, 2)
    draft_logprobs = (0.0, 0.0)
    candidate = SpeculativeDraftCandidate(
        sequence_id="seq-greedy",
        step=1,
        draft_tokens=draft_tokens,
        draft_token_logprobs=draft_logprobs,
    )

    target_logits = np.array(
        [
            [1.0, 5.0, 0.0],  # Argmax is 1 -> Matches draft_tokens[0]
            [8.0, 1.0, 0.0],  # Argmax is 0 -> Differs from draft_tokens[1] (2)
        ],
        dtype=np.float32,
    )

    result = SpeculativeRejectionSampler.verify_and_sample(
        target_logits_batch=target_logits,
        draft_candidate=candidate,
        temperature=0.0,  # Greedy
    )

    # First token accepted, second rejected and corrected to 0
    assert result.accepted_tokens == (1,)
    assert result.correction_token == 0
    assert result.all_emitted_tokens == (1, 0)


def test_edge_draft_speculator_lifecycle() -> None:
    speculator = EdgeDraftSpeculator(sequence_id="seq-spec")
    speculator.set_initial_context([100, 101])
    assert speculator.current_context == (100, 101)

    # Mock next logits predictor function
    def mock_predict(ctx: list[int]) -> NDArray[np.float32]:
        logits = np.zeros(10, dtype=np.float32)
        # Deterministically predict (last_token + 1) % 10
        next_val = (ctx[-1] + 1) % 10
        logits[next_val] = 10.0
        return logits

    draft_candidate = speculator.generate_draft_candidates(
        predict_next_logits_fn=mock_predict,
        horizon_k=3,
        temperature=0.0,
        step=1,
    )

    assert draft_candidate.horizon_length == 3
    # 101 -> 2, 2 -> 3, 3 -> 4
    assert draft_candidate.draft_tokens == (2, 3, 4)

    # Commit verification result: first 2 accepted, 3rd rejected and corrected to 9
    mock_result = SpeculativeVerificationResult(
        sequence_id="seq-spec",
        step=1,
        accepted_tokens=(2, 3),
        correction_token=9,
        num_draft_tokens=3,
    )
    emitted = speculator.commit_verification_result(mock_result)
    assert emitted == (2, 3, 9)
    assert speculator.current_context == (100, 101, 2, 3, 9)


def test_adaptive_horizon_scheduler_scaling() -> None:
    config = SpeculativeConfig(
        default_horizon_k=4,
        min_horizon_k=1,
        max_horizon_k=6,
    )
    scheduler = AdaptiveHorizonScheduler(config)
    assert scheduler.current_horizon_k == 4

    # 1. Simulate high acceptance rate -> horizon expands towards max
    for step in range(10):
        res = SpeculativeVerificationResult(
            sequence_id="seq-high",
            step=step,
            accepted_tokens=(1, 2, 3, 4),
            num_draft_tokens=4,
        )
        scheduler.record_verification(res)

    assert scheduler.current_horizon_k == 6  # Reached max_horizon_k
    assert scheduler.cumulative_acceptance_rate == 1.0
    assert scheduler.estimated_speedup_factor > 2.0

    # 2. Simulate poor acceptance rate -> horizon contracts towards min
    for step in range(15):
        res_poor = SpeculativeVerificationResult(
            sequence_id="seq-low",
            step=step,
            accepted_tokens=(),
            correction_token=0,
            num_draft_tokens=6,
        )
        scheduler.record_verification(res_poor)

    assert scheduler.current_horizon_k == 1  # Contracted down to min_horizon_k
