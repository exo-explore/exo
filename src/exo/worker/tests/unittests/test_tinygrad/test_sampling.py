# pyright: reportUnknownMemberType=false
from tinygrad.tensor import Tensor


def test_sample_result_structure():
    """sample_token should return a SampleResult with token_id, logprob, top_logprobs."""
    from exo.worker.engines.tinygrad.sampling import SampleResult, sample_token

    logits = Tensor([[[0.1, 0.2, 0.9, 0.3]]])
    result = sample_token(logits, temperature=0.0)
    assert isinstance(result, SampleResult)
    assert isinstance(result.token_id, int)
    assert isinstance(result.logprob, float)
    assert isinstance(result.top_logprobs, list)

def test_greedy_sampling():
    """Temperature=0 should return the argmax token."""
    from exo.worker.engines.tinygrad.sampling import sample_token

    logits = Tensor([[[0.1, 0.2, 0.9, 0.3]]])  # token 2 has highest logit
    result = sample_token(logits, temperature=0.0)
    assert result.token_id == 2

def test_temperature_zero_is_deterministic():
    """Greedy sampling must always produce the same result."""
    from exo.worker.engines.tinygrad.sampling import sample_token

    logits = Tensor.randn(1, 1, 1000)
    r1 = sample_token(logits, temperature=0.0)
    r2 = sample_token(logits, temperature=0.0)
    assert r1.token_id == r2.token_id

def test_sampling_returns_valid_token():
    """Sampled token must be in [0, vocab_size)."""
    from exo.worker.engines.tinygrad.sampling import sample_token

    vocab_size = 100
    logits = Tensor.randn(1, 1, vocab_size)
    result = sample_token(logits, temperature=0.7)
    assert 0 <= result.token_id < vocab_size

def test_logprob_is_negative():
    """Log-probabilities must be <= 0 (probabilities are in [0, 1])."""
    from exo.worker.engines.tinygrad.sampling import sample_token

    logits = Tensor.randn(1, 1, 100)
    result = sample_token(logits, temperature=0.7, request_logprobs=True)
    assert result.logprob <= 0.0

def test_greedy_logprob_is_highest():
    """Greedy token should have the highest logprob."""
    from exo.worker.engines.tinygrad.sampling import sample_token

    logits = Tensor([[[0.1, 0.2, 0.9, 0.3]]])
    result = sample_token(logits, temperature=0.0, top_logprobs_count=4, request_logprobs=True)
    # The selected token's logprob should match the top entry
    assert result.token_id == result.top_logprobs[0][0]
    assert abs(result.logprob - result.top_logprobs[0][1]) < 1e-5

def test_top_logprobs_count():
    """top_logprobs should return exactly the requested count."""
    from exo.worker.engines.tinygrad.sampling import sample_token

    logits = Tensor.randn(1, 1, 100)
    result = sample_token(logits, temperature=0.0, top_logprobs_count=5, request_logprobs=True)
    assert len(result.top_logprobs) == 5

def test_top_logprobs_empty_when_not_requested():
    """top_logprobs should be empty list when count is 0."""
    from exo.worker.engines.tinygrad.sampling import sample_token

    logits = Tensor.randn(1, 1, 100)
    result = sample_token(logits, temperature=0.0, top_logprobs_count=0)
    assert result.top_logprobs == []

def test_top_logprobs_sorted_descending():
    """top_logprobs entries should be sorted by logprob (highest first)."""
    from exo.worker.engines.tinygrad.sampling import sample_token

    logits = Tensor.randn(1, 1, 100)
    result = sample_token(logits, temperature=0.0, top_logprobs_count=5, request_logprobs=True)
    logprobs = [lp for _, lp in result.top_logprobs]
    assert logprobs == sorted(logprobs, reverse=True)

def test_high_temperature_is_more_random():
    """Higher temperature should produce more token diversity across runs."""
    from exo.worker.engines.tinygrad.sampling import sample_token

    logits = Tensor.randn(1, 1, 10)
    low_temp_tokens = {sample_token(logits, temperature=0.01).token_id for _ in range(20)}
    high_temp_tokens = {sample_token(logits, temperature=2.0).token_id for _ in range(20)}
    # High temp should generally produce more unique tokens
    # (probabilistic — use a lenient check)
    assert len(high_temp_tokens) >= len(low_temp_tokens)
