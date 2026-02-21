from typing import NamedTuple

from tinygrad.tensor import Tensor


class SampleResult(NamedTuple):
    token_id: int
    logprob: float
    top_logprobs: list[tuple[int, float]]   # (token_id, logprob)

def sample_token(
    logits: Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_logprobs_count: int = 0,
) -> SampleResult:
    last_logits = logits[0, -1, :]

    log_probs = last_logits.log_softmax(axis = -1)

    if temperature == 0:
        token_id = int(last_logits.argmax().item())  # pyright: ignore[reportUnknownMemberType]
    else:
        probs = (last_logits / temperature).softmax(axis = -1)

        r = Tensor.rand(1).item()  # pyright: ignore[reportUnknownMemberType]
        cumsum = probs.cumsum(axis = 0)
        mask = (cumsum > r).float()
        token_id = int(mask.argmax().item())  # pyright: ignore[reportUnknownMemberType]

    selected_logprob = float(log_probs[token_id].item())

    top_logprobs: list[tuple[int, float]] = []
    if top_logprobs_count > 0:
        log_probs_list: list[float] = log_probs.numpy().tolist()  # pyright: ignore[reportAny]
        indexed = sorted(enumerate(log_probs_list), key = lambda x: -x[1])
        top_logprobs= [(tok_id, lp) for tok_id, lp in indexed[:top_logprobs_count]]

    return SampleResult(
        token_id=token_id,
        logprob=selected_logprob,
        top_logprobs=top_logprobs,
    )
